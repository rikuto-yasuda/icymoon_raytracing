# %%
import shutil
import numpy as np
from netCDF4 import Dataset

def sum_density_and_save(src_files, output_file):
    # 初期化
    total_density = None
    dimensions = None

    # 各ファイルから Density 変数を読み込み、足し合わせる
    for file in src_files:
        with Dataset(file, 'r') as nc:
            if 'Density' not in nc.variables:
                raise ValueError(f"'Density' variable not found in file {file}")
            density = nc.variables['Density'][:]
            if total_density is None:
                total_density = np.zeros_like(density)
                dimensions = {dim: nc.dimensions[dim].size for dim in nc.variables['Density'].dimensions}
            else:
                # 次元が一致するか確認
                current_dimensions = {dim: nc.dimensions[dim].size for dim in nc.variables['Density'].dimensions}
                if current_dimensions != dimensions:
                    raise ValueError(f"Dimensions do not match for file {file}")
            total_density += density

    # 最初のファイルをコピーして新しいファイルを作成
    shutil.copy(src_files[0], output_file)

    # 新しい nc ファイルを開き、Density 変数を更新
    with Dataset(output_file, 'a') as nc_out:
        if 'Density' in nc_out.variables:
            density_var = nc_out.variables['Density']
            density_var[:] = total_density*10**6  # 単位を cm^-3 に変換
        else:
            raise ValueError(f"'Density' variable not found in the copied file {output_file}")

        # "phys_length" 変数を1000倍にする
        if 'phys_length' in nc_out.variables:
            phys_length_var = nc_out.variables['phys_length']
            phys_length_var[:] = phys_length_var[:] * 1000  # 単位を変換
        else:
            raise ValueError(f"'phys_length' variable not found in the copied file {output_file}")


    print(f"Summed density saved to {output_file}")

# 使用例
src_files = [
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/python_for_3D_occultation/3d_ionospheric_model/LatHyS_simu/Europa/RUN_A3/H2Opl_19_04_23_t00600.nc',
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/python_for_3D_occultation/3d_ionospheric_model/LatHyS_simu/Europa/RUN_A3/O2pl_19_04_23_t00600.nc',
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/python_for_3D_occultation/3d_ionospheric_model/LatHyS_simu/Europa/RUN_A3/H2pl_19_04_23_t00600.nc',
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/python_for_3D_occultation/3d_ionospheric_model/LatHyS_simu/Europa/RUN_A3/Ojv_19_04_23_t00600.nc'
]
output_file = 'summed_density.nc'
sum_density_and_save(src_files, output_file)

# %%
def print_nc_file_contents(file_path):
    with Dataset(file_path, 'r') as nc:
        print("Dimensions:")
        for dim_name, dim in nc.dimensions.items():
            print(f"  {dim_name}: {len(dim)}")

        print("\nVariables:")
        for var_name, var in nc.variables.items():
            print(f"  {var_name}: {var.dimensions}")
            print(var[:])  # 変数のデータを表示

# 使用例
output_file = 'summed_density.nc'
print_nc_file_contents(output_file)
# %%
