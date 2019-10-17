import numpy as np

BETA_BIAS = {'none':[],
             'distI_0_1':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'distI_0_2':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'distI_1_0':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'distI_2_0':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'distI_1_1':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'distI_2_1':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'distI_1_2':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'distI_2_2':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'opt_c':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'opt_c_I1':np.logspace(np.log10(1), np.log10(2500), num=12, dtype=float),
             'opt_c_I2':np.logspace(np.log10(1), np.log10(2000), num=12, dtype=float),
             'opt_c_s2':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'opt_c_s3':np.logspace(np.log10(1), np.log10(1000), num=12, dtype=float),
             'hs_n1_w05':np.logspace(np.log10(1), np.log10(300), num=12, dtype=float),
             'hs_n1_w10':np.logspace(np.log10(1), np.log10(300), num=12, dtype=float),
             'hs_n1_w15':np.logspace(np.log10(1), np.log10(300), num=12, dtype=float),
             'hs_n1_w20':np.logspace(np.log10(1), np.log10(300), num=12, dtype=float),
             'hs_n2_w05':np.logspace(np.log10(1), np.log10(10000), num=12, dtype=float),
             'hs_n2_w10':np.logspace(np.log10(1), np.log10(10000), num=12, dtype=float),
             'hs_n2_w15':np.logspace(np.log10(1), np.log10(10000), num=12, dtype=float),
             'hs_n2_w20':np.logspace(np.log10(1), np.log10(10000), num=12, dtype=float),
             'hs_n3_w05':np.logspace(np.log10(1), np.log10(500), num=12, dtype=float),
             'hs_n3_w10':np.logspace(np.log10(1), np.log10(500), num=12, dtype=float),
             'hs_n3_w15':np.logspace(np.log10(1), np.log10(500), num=12, dtype=float),
             'hs_n3_w20':np.logspace(np.log10(1), np.log10(500), num=12, dtype=float),
             'hs_r3_w05':np.logspace(np.log10(1), np.log10(178.619), num=11, dtype=float),
             'hs_r3_w10':np.logspace(np.log10(1), np.log10(178.619), num=11, dtype=float),
             'hs_r3_w15':np.logspace(np.log10(1), np.log10(178.619), num=11, dtype=float),
             'hs_r3_w20':np.logspace(np.log10(1), np.log10(178.619), num=11, dtype=float)}

for r in [0., 0.5, 1., 2.]:
    for w in [5,10,15,20]:
        end = 100 + 50 * (r + w/5.)
        bias = np.logspace(np.log10(100), np.log10(end), num=5, dtype=float)
        BETA_BIAS.update({f"im5_r{r:3.1f}_w{w:02d}": bias})

for w in [5,10,15,20]:
#   start = 0.8 * (100 + (50 * w / 5.))
#   end = 1.5 * (100 + (50 * w / 5.))
#   bias = np.logspace(np.log10(start), np.log10(end), num=10, dtype=float)

#   bias = np.array([1, 2, 4, 8, 14, 20, 28, 35, 43, 52, 60, 69, 78, 88, 99, 110, 122, 135, 150, 170, 195, 220, 250])
#   bias = np.array([130, 140, 160, 180, 210, 240, 270, 300])
    bias = np.array([200, 240, 280])
    BETA_BIAS.update({f"Nhs_n1_w{w:02d}": bias})

    bias = np.array([30, 35, 40, 50, 55, 60])
    BETA_BIAS.update({f"Nhs_n2_w{w:02d}": bias})

    bias = np.array([10, 15, 20, 25, 30, 35])
    BETA_BIAS.update({f"Nhs_n3_w{w:02d}": bias})


    BETA_BIAS['distI_1_0'] = np.array([0.5, 1, 1.5, 2.1, 2.7, 3.3, 4.0, 5, 6, 7, 8, 10, 12, 14, 16, 20])*250
    BETA_BIAS['distI_2_0'] = np.array([0.5, 1, 1.5, 2.1, 2.7, 3.3, 4.0, 5, 6, 7, 8, 10, 12, 14, 16, 20])*1000
    BETA_BIAS['distI_3_0'] = np.array([0.5, 1, 1.5, 2.1, 2.7, 3.3, 4.0, 5, 6, 7, 8, 10, 12, 14, 16, 20])*2000

    bias = np.array([50, 200, 500, 750, 1000, 1250, 1500, 1750, 2000])
    BETA_BIAS.update({"distI_3_0": bias})



    bias = np.array([1, 3, 6, 9, 12, 15, 18, 22, 26, 30, 35, 40, 50, 60, 80, 100, 120, 140,160,180,200, 220, 240, 260])
    BETA_BIAS.update({f"TRANSB_1": bias})

    bias = np.array([1, 3, 6, 10, 14, 18, 22, 26, 30, 35, 40, 60, 80, 100, 120, 140,160,180,200, 220, 240, 260])
    BETA_BIAS.update({f"TRANSB_2": bias})

    bias = np.array([1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100, 120, 140,160,180,200])
    BETA_BIAS.update({f"TRANSB_3": bias})



#   start = 80
#   end = 200 
#   bias = np.logspace(np.log10(start), np.log10(end), num=10, dtype=float)

#   bias = np.array([33, 45, 55, 65, 75, 85, 95, 105, 115, 128, 140])
    bias = np.array([40, 50, 60])
    BETA_BIAS.update({f"Nim5_r0.0_w{w:02d}": bias})

