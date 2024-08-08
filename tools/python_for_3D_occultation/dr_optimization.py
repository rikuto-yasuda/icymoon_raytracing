# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
HIG = "0" # -2500 -1500 -500 0 [km]
FREQ = "5.644999980926513672e6" # 3.984813988208770752e5 5.644999980926513672e6 [Hz]
STEP_LEN = "100e3"# 0.1e3 1e3 10e3 100e3 1000e3 [m]
PREC = "0.0000001" # 1 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.000001 [ratio]


# %%
FREQ_array = ["3.984813988208770752e5", "5.644999980926513672e6"]
HIG_array = [ "0","-500", "-1500", "-2500"]
STEP_LEN_array = ["0.1e3", "1e3", "10e3", "100e3", "1000e3"]
# PREC_array = ["1", "0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001", "0.0000001"]
PREC_array = ["0.0001","0.00009","0.00008","0.00007","0.00006","0.00005","0.00004","0.00003","0.00002","0.00001","0.000009","0.000008","0.000007","0.000006","0.000005","0.000004","0.000003","0.000002","0.000001","0.0000009","0.0000008","0.0000007","0.0000006","0.0000005","0.0000004","0.0000003","0.0000002","0.0000001"]
HIG_array = [ "0"]


# %%

# Check PREC variation
for freq in range(len(FREQ_array)):
    FREQ = FREQ_array[freq]
    for hig in range(len(HIG_array)):
        HIG = HIG_array[hig]
        for step in range(len(STEP_LEN_array)):
            STEP_LEN = STEP_LEN_array[step]
            for prec in range(len(PREC_array)):
                PREC = PREC_array[prec]
                print("PREC: ", PREC)
                print("/Users/yasudarikuto/research/icymoon_raytracing/src_venv/rtc_cost_reduction/testing/ray-Pganymede_nonplume_100e2_0.1e2-Mtest_simple-benchmark-LO-Z"+HIG+"-FR"+FREQ+"-STEP"+STEP_LEN+"-PREC"+PREC)
                data = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/src_venv/rtc_cost_reduction/testing/ray-Pganymede_nonplume_100e2_0.1e2-Mtest_simple-benchmark-LO-Z"+HIG+"-FR"+FREQ+"-STEP"+STEP_LEN+"-PREC"+PREC)
                if data.shape[0] == 14 or data.shape[0] == 0:
                    continue
                ray_x = data[:, [1]]
                ray_y = data[:, [2]]
                ray_z = data[:, [3]]
                plt.plot(ray_x, ray_z , label="PREC"+str(PREC))

            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.title("PREC variation"+", HIG: "+HIG+", FREQ: "+FREQ+", STEP_LEN: "+STEP_LEN)

            if HIG == "-500" and FREQ == "3.984813988208770752e5":
                plt.xlim(600, 800) # ok
                plt.ylim(6800, 7000)

            elif HIG == "-500" and FREQ == "5.644999980926513672e6":
                plt.xlim(-1600, -1400) # ok
                plt.ylim(-600, -400)

            elif HIG == "-1500" and FREQ == "3.984813988208770752e5":
                plt.xlim(-8000, -7800) # ok
                plt.ylim(5200, 5400)

            elif HIG == "-1500" and FREQ == "5.644999980926513672e6":
                plt.xlim(-2500, -2300)
                plt.ylim(-1600, -1400)

            elif HIG == "-2500" and FREQ == "3.984813988208770752e5":
                plt.xlim(-11800, -11600) # ok
                plt.ylim(-1600, -1400)

            elif HIG == "-2500" and FREQ == "5.644999980926513672e6":
                plt.xlim(-2700, -2500)
                plt.ylim(-2600, -2400)

            
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            plt.savefig("/Users/yasudarikuto/research/icymoon_raytracing/tools/python_for_3D_occultation/dr_optimazation_results/raypath/HIG: "+HIG+", FREQ: "+FREQ+", STEP_LEN: "+STEP_LEN+".png")
            plt.show()


# %%
for freq in range(len(FREQ_array)):
    FREQ = FREQ_array[freq]
    for hig in range(len(HIG_array)):
        HIG = HIG_array[hig]
        for step in range(len(STEP_LEN_array)):
            STEP_LEN = STEP_LEN_array[step]
            theta =0
            for prec in range(len(PREC_array)):
                PREC = PREC_array[prec]
                print("PREC: ", PREC)
                print("HIG: ", HIG)
                print("FREQ: ", FREQ)
                print("STEP_LEN: ", STEP_LEN)  

                data = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/src_venv/rtc_cost_reduction/testing/ray-Pganymede_nonplume_100e2_0.1e2-Mtest_simple-benchmark-LO-Z"+HIG+"-FR"+FREQ+"-STEP"+STEP_LEN+"-PREC"+PREC)
                print(data.shape)                  
                if data.shape[0] == 14 or data.shape[0] == 0:
                    continue
                ray_x = data[:, [1]]
                ray_y = data[:, [2]]
                ray_z = data[:, [3]]

                ray_r = np.sqrt(ray_x**2 + ray_z**2)

                later_pos = np.where(ray_x > -3000)[0]
                if len(later_pos) == 0:
                    continue
                print("later pos:" + str(later_pos[0]))
                later_array = np.arange(int(later_pos[0]),len(ray_x))
                print("later_array: ", later_array)

                boundary_outide_ind = np.where(ray_r > 6501)[0]
                print("boundary_outide_ind: ", boundary_outide_ind)

                shared_elements = np.intersect1d(later_array, boundary_outide_ind)
                print("shared_elements: ", shared_elements)

                if len(shared_elements) == 0:
                    continue

                boundary_outside_ind = shared_elements[0]

                boundary_inside_ind = boundary_outside_ind - 1

                boundary_inside_x = ray_x[boundary_inside_ind]
                boundary_inside_z = ray_z[boundary_inside_ind]

                boundary_outside_x = ray_x[boundary_outside_ind]
                boundary_outside_z = ray_z[boundary_outside_ind]

                boundary_inside_r = ray_r[boundary_inside_ind]
                boundary_outside_r = ray_r[boundary_outside_ind]

                boundary_ratio = (6501 - boundary_inside_r) / (boundary_outside_r - boundary_inside_r)

                boundary_x = boundary_inside_x + boundary_ratio * (boundary_outside_x - boundary_inside_x)
                boundary_z = boundary_inside_z + boundary_ratio * (boundary_outside_z - boundary_inside_z)

                theta = np.rad2deg(np.arctan(boundary_z / boundary_x))
                plt.scatter(float(PREC), theta, color='black')

                # print("theta: ", theta)
            deg_error = np.rad2deg(np.arctan(1/6501))
            if theta !=0:
                plt.axhline(y=deg_error+theta, color='r', linestyle='--', label="1km")
                plt.axhline(y=-1*deg_error+theta, color='r', linestyle='--')
                plt.axhline(y=2*deg_error+theta, color='b', linestyle='--', label="2km")
                plt.axhline(y=-2*deg_error+theta, color='b', linestyle='--')
                plt.axhline(y=3*deg_error+theta, color='g', linestyle='--', label="3km")
                plt.axhline(y=-3*deg_error+theta, color='g', linestyle='--')


            plt.xlabel("PREC")
            plt.ylabel("theta [deg]")
            plt.xscale('log')
            plt.title("Error "+", HIG: "+HIG+", FREQ: "+FREQ+", STEP_LEN: "+STEP_LEN)
            plt.legend()
            plt.savefig("/Users/yasudarikuto/research/icymoon_raytracing/tools/python_for_3D_occultation/dr_optimazation_results/Error/Error"+", HIG: "+HIG+", FREQ: "+FREQ+", STEP_LEN: "+STEP_LEN+".png")
            plt.show()

# %%
for prec in range(len(PREC_array)):
    PREC = PREC_array[prec]
    for step in range(len(STEP_LEN_array)):
        STEP_LEN = STEP_LEN_array[step]
        print("STEP_LEN: ", STEP_LEN)
        data = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/src_venv/rtc_cost_reduction/testing/ray-Pganymede_nonplume_100e2_0.1e2-Mtest_simple-benchmark-LO-Z"+HIG+"-FR"+FREQ+"-STEP"+STEP_LEN+"-PREC"+PREC)
        if data.shape[0] == 14 or data.shape[0] == 0:
            continue
        ray_x = data[:, [1]]
        ray_y = data[:, [2]]
        ray_z = data[:, [3]]

        plt.plot(ray_x, ray_z , label="STEP_LEN"+str(STEP_LEN))

    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("STEP_LEN variation"+", HIG: "+HIG+", FREQ: "+FREQ+", PREC: "+PREC)
    plt.legend()
    plt.show()
# %%
