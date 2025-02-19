'''
Author: lizd lizd@ios.ac.cn
Date: 2025-02-11 15:10:30
LastEditors: lizd lizd@ios.ac.cn
LastEditTime: 2025-02-18 13:17:02
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


if __name__ == "__main__":
    labels_out = read_grid_file("./data/zcurve_wnf.bin")
    view_data(labels_out)


