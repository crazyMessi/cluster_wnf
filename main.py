'''
Author: lizd lizd@ios.ac.cn
Date: 2025-02-20 15:02:10
LastEditors: lizd lizd@ios.ac.cn
LastEditTime: 2025-02-25 16:12:11
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

model_name = 'scene0001_57dist.ply'

def parse_args():
    parser = argparse.ArgumentParser(description='Cluster WNF')
    parser.add_argument('--input', type=str, default='./data/input/'+model_name, help='input point cloud file')
    parser.add_argument('--output', type=str, default='./data/output/'+model_name, help='output point cloud file')
    parser.add_argument('--resolution', type=int, default=64, help='resolution of the grid')
    parser.add_argument('--compactness', type=float, default=0.1, help='compactness of the grid')
    parser.add_argument('--k', type=int, default=10, help='k of the grid')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon of the grid')
    parser.add_argument('--n_segments', type=int, default=2000, help='n_segments of the grid')
    parser.add_argument('--iters', type=int, default=10, help='iters')
    return parser.parse_args()


def save_op(points,normals,path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path, pcd)

def save_op_mesh(verts,faces,path):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(path, mesh)

def main():
    args = parse_args()
    print(args)
    # 读取点云
    pcd = o3d.io.read_point_cloud(args.input)
    ori_points = np.asarray(pcd.points)
    normalized_points, iXForm = tools.transform_points(ori_points)
    # normals = tools.PCA_normal_estimate(normalized_points)
    normals = np.asarray(pcd.normals)
    
    pcd.points = o3d.utility.Vector3dVector(normalized_points)
    # 使用泊松重建得到GT poisson 曲面
    gt_mesh,val = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    gt_mesh = tools.clean_mesh(gt_mesh,pcd)
    gt_verts = np.asarray(gt_mesh.vertices)
    gt_faces = np.asarray(gt_mesh.triangles)
    
    
    # 计算winding number
    bbox = np.array([[-1,1],[-1,1],[-1,1]])
    wnf_calculator = cal_wnf.WNF(points=normalized_points,epsilon=args.epsilon)
    wnf_calculator.update_normal(normals)
    grid, grid_shape = ncut.create_uniform_grid(args.resolution,bbox)

    # wnf_field = wnf_calculator.query_wn(grid)
    # wnf_field.clamp_(-1,1)
    # wnf_field = np.reshape(wnf_field.cpu().numpy(), grid_shape)
    # view.view_data_and_mesh(wnf_field,gt_verts,gt_faces,bbox)

    # normals = tools.PCA_normal_estimate(normalized_points)
    # wnf_calculator.update_normal(normals)
    for i in range(args.iters):
        wnf_field = wnf_calculator.query_wn(grid)
        wnf_field.clamp_(-100,100)

        # 计算超体素网络
        wnf_field = np.reshape(wnf_field.cpu().numpy(), grid_shape)
        
        # clusters,_,_,_ = ncut.field_to_clusters_with_vis(wnf_field, args.n_segments, args.compactness, "temp/iter_{}".format(i))
        clusters,segmented_grid,_,_ = ncut.field_to_clusters(wnf_field, args.n_segments, args.compactness)
        # 提取面片
        verts, faces = tools.extract_surface_from_scalar_field(clusters, 0, args.resolution,bbox)
        # verts, faces = tools.clean_mesh2(verts,faces,normalized_points)
        
        # 更新法向量
        normals = update_normal.compute_point_normals_from_mesh(normalized_points,verts, faces, args.k)
        wnf_calculator.update_normal(normals)
        
        save_op(normalized_points,normals,"temp/iter_{}.ply".format(i))
        save_op_mesh(verts,faces,"temp/iter_{}_mesh.ply".format(i))
        
        slice_idx = (19,19,19)
        tools.plot_partition(clusters,"temp/iter_{}_clusters.png".format(i),slice_idx)
        tools.plot_partition(wnf_field,"temp/iter_{}_wnf.png".format(i),slice_idx)

        view.view_data_and_mesh(segmented_grid,gt_verts,gt_faces,bbox)
        view.view_data_and_mesh(clusters,verts,faces,bbox)

        print(f"iter {i} done") 

if __name__ == "__main__":
    main()