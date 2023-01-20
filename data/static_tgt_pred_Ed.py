import pprint, os
from utils.utils import read_json, get_best_formula
from mab_ml.data.data import id_hull_json
from pymatgen.io.vasp.inputs import Structure


import matplotlib.pyplot as plt
from PIL import Image

tgt_pred = read_json(r'best_tgt_pred.json') # from mab_ml\check_log.py
infos = id_hull_json() # from yqsun_materials_project_lib\convex_hull_builder\convex.py
# print(tgt_pred)

phonolabel = read_json(r'G:\high-throughput-workflow\allthrough_heatmap\phononlabels')
print(len(phonolabel))

for id, pred in tgt_pred.items():
    infos[id]['predict'] = pred['predict']
# print(infos)

tgt_dic = {}
for id, dic in infos.items():
    tgt = dic['hull_distance']
    pred = dic['predict']
    form = dic["pretty_formula"]
    fdir = r'G:\codes\modnet\mab_ml\data\mp_opted_structures'
    fname  = form+'_212_orth_mp.vasp'
    path = os.path.join(fdir, fname)
    structure = Structure.from_file(path)
    formula = get_best_formula(structure)
    if formula in phonolabel.keys():
        dynamic_state = phonolabel[formula]
    else:
        dynamic_state = 'u'

    tgt_dic[id] = [form, formula, tgt, pred, abs(tgt-pred)]

num = len(tgt_dic)
print(num)
# pprint.pprint(tgt_dic)




# stable = []
# metastable = []
# unstabel = []
# for ls in tgt_dic.values(): # [form, formula, tgt, pred, abs(tgt-pred)]
#     try:
#         if ls[1] in phonolabel.keys():
#             state = phonolabel[ls[1]]
#         if state == 's':
#             all.append(ls[1])
#             # if 'Al' not in ls[1]:
#             #     no_Al.append(ls[1])
#     except:
#         checks.append(ls[1])
# print(all)
# print(no_Al)
# print(checks)
# print(len(all))
# print(len(no_Al))
# print(len(checks))
#
# ### ploootttt
# spectrum_dir = r'C:\Users\sunyuqi\Desktop\new_manual_spectrum\manual_spectrum'
# sp_list = os.listdir(spectrum_dir)
# sp_list = [os.path.join(spectrum_dir, s) for s in sp_list]



# stable_list = []
# for label, mab in zip(phonolabel.values(), phonolabel.keys()):
#     if label == 's':
#         stable_list.append(mab)
# print(stable_list)

stable_list = [
    ['Sc2ZnB2', 'Ti2ZnB2', 'Zr2ZnB2', 'Hf2ZnB2', 'V2ZnB2',
     'Nb2ZnB2', 'Ta2ZnB2', 'Cr2ZnB2', 'Sc2CdB2', 'Y2CdB2',
     'Ti2CdB2', 'Zr2CdB2', 'Hf2CdB2', 'V2CdB2', 'Nb2CdB2', ],
    ['Ta2CdB2', 'Sc2AlB2', 'Ti2AlB2', 'Zr2AlB2', 'Hf2AlB2',
     'V2AlB2', 'Nb2AlB2', 'Ta2AlB2', 'Cr2AlB2', 'Mo2AlB2',
     'W2AlB2', 'Mn2AlB2', 'Tc2AlB2', 'Fe2AlB2', 'Co2AlB2', ],
    ['Ni2AlB2', 'Sc2GaB2', 'Y2GaB2', 'Ti2GaB2', 'Zr2GaB2',
     'Hf2GaB2', 'V2GaB2', 'Nb2GaB2', 'Ta2GaB2', 'Cr2GaB2',
     'W2GaB2', 'Mn2GaB2', 'Fe2GaB2', 'Co2GaB2', 'Sc2InB2', ],
    ['Y2InB2', 'Ti2InB2', 'Zr2InB2', 'Hf2InB2', 'V2InB2',
     'Nb2InB2', 'Ta2InB2', 'Sc2TlB2', 'Y2TlB2', 'Ti2TlB2',
     'Zr2TlB2', 'Hf2TlB2', 'Sc2SiB2', 'Ti2SiB2', 'Cr2SiB2', ],
    ['Mn2SiB2', 'Fe2SiB2', 'Sc2GeB2', 'Y2GeB2', 'Mn2GeB2',
     'Sc2SnB2', 'Cr2PB2', 'Mo2PB2', 'Cr2AsB2', 'Mo2AsB2']
]

# for plot_mab_list in stable_list:
#     count = 0
#     # fig = plt.figure(30, 50)
#     for mab in plot_mab_list:
#         for sp in sp_list:
#             if mab in sp:
#                 # print("Open a mab phonon spectrum png ...")
#                 img = Image.open(sp)
#                 count += 1
#                 plt.suptitle('212-MAB phonon spectrum')  # 图片名称
#                 plt.subplot(3, 5, count)
#                 plt.title(mab)
#                 plt.imshow(img)
#                 plt.axis('off')
#     plt.savefig('subplot_phonon' + str(stable_list.index(plot_mab_list)))