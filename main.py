'''
Author: lizd lizd@ios.ac.cn
Date: 2025-02-20 15:02:10
LastEditors: lizd lizd@ios.ac.cn
LastEditTime: 2025-02-26 15:52:55
FilePath: \cluster_wnf\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import open3d as o3d
import os
import sys
import cal_wnf
import ncut
import update_normal
import argparse 
import tools
import view
import torch
from sklearn.cluster import SpectralClustering
import cc3d
import optimization
import cluster

model_name = 'planck_multisample_n2000_input.ply'

def parse_args():
    parser = argparse.ArgumentParser(description='Cluster WNF')
    parser.add_argument('--input', type=str, default='./data/input/'+model_name, help='input point cloud file')
    parser.add_argument('--output', type=str, default='./data/output/'+model_name, help='output point cloud file')
    parser.add_argument('--resolution', type=int, default=128, help='resolution of the grid')
    parser.add_argument('--compactness', type=float, default=0.1, help='compactness of the grid')
    parser.add_argument('--k', type=int, default=10, help='k of the grid')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon of the grid')
    parser.add_argument('--n_segments', type=int, default=2000, help='n_segments of the grid')
    parser.add_argument('--iters', type=int, default=10, help='iters')
    parser.add_argument('--distance', type=float, default=0.03, help='size of the double cover')
    return parser.parse_args()


def save_op(points,normals,path):
    if not os.path.exists(os.path.dirname(path)):
        tools.rmdir(os.path.dirname(path))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path, pcd)

def save_op_mesh(verts,faces,path):
    if not os.path.exists(os.path.dirname(path)):
        tools.rmdir(os.path.dirname(path))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(path, mesh)

def main(input='./data/input/'+model_name,output='./data/output/'+model_name,resolution=128 ,compactness=0.1,k=10,epsilon=1e-8,n_segments=2000,iters=10,distance=0.1):
   
    # 读取点云
    pcd = o3d.io.read_point_cloud(input)
    ori_points = np.asarray(pcd.points)
    normalized_points, iXForm = tools.transform_points(ori_points)
    gt_normals = np.asarray(pcd.normals)
    if gt_normals.shape[0] == 0:
        gt_normals = tools.pymeshlab_normal_estimate(normalized_points,10)
        print("no gt normals, use pymeshlab to estimate")
    normalized_points,gt_normals = tools.clean_bad_data(normalized_points,gt_normals)
    
    pcd.points = o3d.utility.Vector3dVector(normalized_points)
    pcd.normals = o3d.utility.Vector3dVector(gt_normals)
    # 使用泊松重建得到GT poisson 曲面
    gt_mesh,val = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    gt_mesh = tools.clean_mesh(gt_mesh,pcd)
    gt_verts = np.asarray(gt_mesh.vertices)
    gt_faces = np.asarray(gt_mesh.triangles)
    # 保存gt mesh
    save_op_mesh(gt_verts,gt_faces,"temp/gt_mesh.ply")
    
    
    # 计算winding number
    bbox = np.array([[-1,1],[-1,1],[-1,1]])
    wnf_calculator = cal_wnf.WNF(points=normalized_points,epsilon=epsilon)
    grid, grid_shape = tools.create_uniform_grid(resolution,bbox)
    
    if distance > 0:
        maskk = (distance * resolution * 2) ** 3
        maskk = int(maskk)
        mask = tools.create_mask_by_k(grid,normalized_points,maskk)
        # smask = tools.create_mask_by_k(grid,normalized_points,maskk-1)
    else:
        mask = np.ones(grid.shape[0],dtype=bool)
        # smask = np.ones(grid.shape[0],dtype=bool)
    mask = mask.reshape(grid_shape)
    
    # 使用cc3d判断mask是否联通
    labels = cc3d.connected_components(mask.reshape(grid_shape))
    if labels.max() > 1:
        print("mask is not connected, try to increase distance")
        exit()

    slice_idx = (args.resolution//2,args.resolution//2,args.resolution//2)
    for iteration in range(iters):
        if iteration == 0:
            wnf_calculator.update_normal(gt_normals)
            gt_cluster = None
        else:
            # normals = tools.PCA_normal_estimate(normalized_points)
            normals = tools.pymeshlab_normal_estimate(normalized_points,k=10)
            wnf_calculator.update_normal(normals)

        wnf_field = torch.zeros(grid.shape[0],dtype=torch.float32,device=wnf_calculator.device)
        wnf_field[mask.flatten()] = wnf_calculator.query_wn(grid[mask.flatten()])
        wnf_field.clamp_(-1,1)

        points_count = ncut.get_point_counts_per_grid(normalized_points,grid_shape,bbox)
        points_count = np.reshape(points_count,grid_shape)

        wnf_field = np.reshape(wnf_field.cpu().numpy(), grid_shape)
        segmented_grid = ncut.segment_winding_field(wnf_field, n_segments, compactness, mask=mask.reshape(grid_shape)).flatten()        
        # segmented_grid = ncut.cc3d_segmentatino(wnf_field,level_count=n_segments,min_size=100)
        segmented_grid = np.reshape(segmented_grid,grid_shape) 
        segmented_grid[mask],segment_mean = ncut.reallcate_labels_by_mean_val(segmented_grid[mask],wnf_field[mask])
        mask = mask.reshape(grid_shape)
        
        tree_cut_res = cluster.tree_cut(segmented_grid,wnf_field,points_count,mask)
        
        cluster_grid = cluster.max_cut(segmented_grid,wnf_field,points_count,mask)
        tools.plot_partition(cluster_grid,"temp/{}/iter_{}_clusters_maxcut.png".format(model_name,iteration),slice_idx,bbox,normalized_points)
        cluster_grid = cluster.morphological_subdivision_with_dilation_3d(cluster_grid,tools.get_kernel_correspond_to_connectivity(6),iterations=2)
        tools.plot_partition(cluster_grid,"temp/{}/iter_{}_clusters_maxcut_morphological.png".format(model_name,iteration),slice_idx,bbox,normalized_points)
        cluster_grid[mask],cluster_mean = ncut.reallcate_labels_by_mean_val(cluster_grid[mask].flatten(),wnf_field[mask].flatten())
        cluster_grid = cluster.max_cut(cluster_grid,wnf_field,points_count,mask)
        tools.plot_partition(cluster_grid,"temp/{}/iter_{}_clusters_maxcut_morphological_maxcut.png".format(model_name,iteration),slice_idx,bbox,normalized_points)
        
        # 提取面片
        # 
        verts, faces = tools.extract_surface_from_scalar_field(cluster_grid, 0.5, resolution,bbox,mask=mask.reshape(grid_shape))
        # verts, faces = tools.clean_mesh2(verts,faces,normalized_points)
        
        # 更新法向量
        ori_normals = wnf_calculator.normals.cpu().numpy()
        normals = update_normal.compute_point_normals_from_mesh(normalized_points,verts, faces, k)
        avg_angle_diff = np.mean(np.abs(np.sum(ori_normals*normals,axis=1)))
        print('avg_angle_diff: %g' % avg_angle_diff)
        # normals = update_normal.dir_update_normal(normalized_points,verts,faces,k)
        
        
        wnf_calculator.update_normal(normals)
        
        save_op(normalized_points,normals,"temp/{}/iter_{}_normals.ply".format(model_name,iteration))
        save_op_mesh(verts,faces,"temp/{}/iter_{}_mesh.ply".format(model_name,iteration))
        
        slice_idx = (args.resolution//2,args.resolution//2,args.resolution//2)
        tools.plot_partition(cluster_grid,"temp/{}/iter_{}_clusters.png".format(model_name,iteration),slice_idx,bbox,normalized_points)
        tools.plot_partition(segmented_grid,"temp/{}/iter_{}_segments.png".format(model_name,iteration),slice_idx,bbox,normalized_points)
        tools.plot_partition(wnf_field,"temp/{}/iter_{}_wnf.png".format(model_name,iteration),slice_idx,bbox,normalized_points)

        # 保存wnf_field,segmented_grid,cluster_grid
        np.save("temp/{}/iter_{}_wnf.npy".format(model_name,iteration),wnf_field)
        np.save("temp/{}/iter_{}_segmented_grid.npy".format(model_name,iteration),segmented_grid)
        np.save("temp/{}/iter_{}_cluster_grid.npy".format(model_name,iteration),cluster_grid)
        
        # view.view_data_and_mesh(segmented_grid,gt_verts,gt_faces,bbox)
        # view.view_data_and_mesh(wnf_field,verts,faces,bbox)

        print(f"iter {iteration} done") 
    return wnf_field,segmented_grid,cluster_grid

if __name__ == "__main__":
    args = parse_args()
    print(args)
    if "/" in args.input:
        model_name = args.input.split('/')[-1].split('.')[0]
    elif "\\" in args.input:
        model_name = args.input.split('\\')[-1].split('.')[0]
    
    if not os.path.exists(os.path.dirname("temp/{}/".format(model_name))):
        tools.rmkdir("temp/{}/".format(model_name))
    wnf_field,segmented_grid,cluster_grid = main(args.input,args.output,args.resolution,args.compactness,args.k,args.epsilon,args.n_segments,args.iters,args.distance)