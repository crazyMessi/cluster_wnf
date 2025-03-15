'''
Author: lizd lizd@ios.ac.cn
Date: 2025-02-11 15:10:30
LastEditors: lizd lizd@ios.ac.cn
LastEditTime: 2025-02-21 18:17:29
FilePath: \cluster_wnf\view.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import polyscope as ps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def read_grid_file(filename):
    with open(filename, 'rb') as f:
        # 读取分辨率（一个 int 值）
        res = np.fromfile(f, dtype=np.int32, count=1)[0]

        # 读取浮点数数据
        total_floats = res * res * res
        grid_data = np.fromfile(f, dtype=np.float32, count=total_floats)

        # 将数据重塑为三维数组
        grid_data = grid_data.reshape((res, res, res))

    return grid_data


def view_data(labels_out,bound_low=[0,0,0],bound_high = [1,1,1]):
    ps.init()
    # define the resolution and bounds of the grid

    # color = cm.get_cmap("Paired")(labels_out).astype(float)[...,:3].reshape((-1,3))
    # register the grid
    ps_grid = ps.register_volume_grid("sample grid", labels_out.shape, bound_low, bound_high)


    # add a scalar function on the grid
    ps_grid.add_scalar_quantity("node scalar1", labels_out,
                                defined_on='nodes', enabled=True)

    ps.show()

def view_data_and_mesh(labels_out,verts,faces,bbox):
    bound_low = [bbox[0,0],bbox[1,0],bbox[2,0]]
    bound_high = [bbox[0,1],bbox[1,1],bbox[2,1]]
    ps.init()
    # 不需要对颜色进行插值
    ps_grid = ps.register_volume_grid("sample grid", labels_out.shape, bound_low, bound_high)
    
    
    ps_grid.add_scalar_quantity("node scalar1", labels_out,
                                defined_on='nodes', enabled=True)
    ps.register_surface_mesh("sample mesh", verts, faces)
    ps.show()

def view_data_and_points(labels_out,points,bbox):
    bound_low = [bbox[0,0],bbox[1,0],bbox[2,0]]
    bound_high = [bbox[0,1],bbox[1,1],bbox[2,1]]
    ps.init()
    ps_grid = ps.register_volume_grid("sample grid", labels_out.shape, bound_low, bound_high)
    ps_grid.add_scalar_quantity("node scalar1", labels_out, defined_on='nodes', enabled=True)
    ps.register_point_cloud("sample points", points)
    ps.show()



if __name__ == "__main__":
    from argparse import ArgumentParser
    from gooey import Gooey, GooeyParser

    @Gooey(program_name="View WNF Field", default_size=(800, 600))
    def main():
        parser = GooeyParser(description="View WNF Field")
        parser.add_argument("Input", widget="FileChooser", help="Input WNF Field File")
        args = parser.parse_args()
        if ".bin" in args.Input:
            labels_out = read_grid_file(args.Input)
        elif ".npy" in args.Input:
            labels_out = np.load(args.Input)
        view_data(labels_out)
        
    main()

