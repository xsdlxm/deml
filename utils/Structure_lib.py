import os, re
from pathlib import Path
from pymatgen.io.vasp.inputs import Structure

def get_structure_paths(dir):
    structures_list = os.listdir(dir)
    paths = [os.path.join(dir, i) for i in structures_list]
    return paths

def sort_ele_list(ele_list):
    rule = {'Sc': 0, 'Y': 1, 'Ti': 2, 'Zr': 3, 'Hf': 4, 'V': 5, 'Nb': 6, 'Ta': 7, 'Cr': 8, 'Mo': 9, 'W': 10,
            'Mn': 11, 'Tc': 12, 'Fe': 13, 'Ru': 14, 'Co': 15, 'Rh': 16, 'Ni': 17}
    new_list = []
    for i in rule:
        # print(i)
        for k in ele_list:
            if k['element'] == i:
                new_list.append(k)
    return new_list

class stlist():
    def __init__(self, dir):
        self.dir = dir

    def file_name(self, file_dir):
        for root, dirs, files in os.walk(file_dir):
            return files

    def get_stlist(self):
        files = self.file_name(self.dir)
        st_list = []
        for i in files:
            j = os.path.join(self.dir, i)
            st_list.append(j)
        # st_list.pop()
        return st_list

def get_best_formula(structure):
    structure = structure.get_primitive_structure()
    formula = structure.composition.formula
    # print(formula)
    a = formula.split(' ')
    b = [1, 2, 3]
    for i in a:
        if '1' in i:
            b[1] = i.strip('1')
        if 'B2' in i:
            b[2] = i
        if '2' in i and 'B' not in i:
            b[0] = i
    return b[0] + b[1] + b[2]
    # modi_formula = a[0] + a[1] + a[2]
    # return modi_formula

class dirlist():
    def __init__(self, dir):
        self.dir = dir

    def file_name(self, file_dir):
        for root, dirs, files in os.walk(file_dir):
            return dirs

    def get_stlist(self):
        dirs = self.file_name(self.dir)
        st_list = []
        for i in dirs:
            j = os.path.join(self.dir, i)
            st_list.append(j)
        # st_list.pop()
        return st_list

def rename_dir_by_formula(target_dir):
    target_dir_list = dirlist(dir=target_dir).get_stlist()
    print(target_dir_list)
    for i in target_dir_list:
        path = Path(i)
        # print(path)
        if path.is_dir() != False:
            pathname = Path(i).name
            # print(path)
            if 'launch' in pathname:
                struct = Structure.from_file(os.path.join(i, 'POSCAR'))
                bestformula = get_best_formula(struct)
                root = Path(i).parent
                print(root)
                Path(i).rename(os.path.join(root, bestformula))


class elements():
    def __init__(self,
                 space_group = 'hex',
                 M_plan_str="Sc, Y, Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W, Mn, Tc, Fe, Ru, Co, Rh, Ni",
                 A_plan_str="Zn, Cd, Al, Ga, In, Tl, Si, Ge, Sn, Pb, P, As, S", is_print=False):
        self.space_group = space_group
        self.M_plan_str = M_plan_str
        self.A_plan_str = A_plan_str
        self.is_print = is_print
    def M(self):
        M_plan_list = self.M_plan_str.split(", ")

        if self.is_print:
            print('There are %s M elements: ' % str(len(M_plan_list)))
            print(self.M_plan_str)
        return M_plan_list

    def A(self):
        A_plan_list = self.A_plan_str.split(", ")

        if self.is_print:
            print('There are %s A elements: ' % str(len(A_plan_list)))
            print(self.A_plan_str)
        return A_plan_list


class ele_num():
    def __init__(self, structure):
        self.structure = structure
    
    def get_atom_number_dict(self):
        # get MAB elements and atom number
        formula_sp = self.structure.formula.split()
        ele_dict = {}
        for i in formula_sp:
            num = re.findall('(\d+)', i)[0]
            # print(num)
            ele = i.split(num)
            ele = ele[0]
            # print(ele)
            ele_dict[ele] = int(num)
        return ele_dict
    
    def get_M_ele(self):
            _M_ele = ''
            for i in self.get_atom_number_dict().keys():
                if i in elements().M():
                    _M_ele += i
            return _M_ele
    
    def get_A_ele(self):
            _A_ele = ''
            for i in self.get_atom_number_dict().keys():
                if i in elements().A():
                    _A_ele += i
            return _A_ele

if __name__=='__main__':
    # structure = Structure.from_file(r'G:\high-throughput-workflow\332_opt\0\POSCAR')
    # best = getbestformula(structure.get_primitive_structure())
    # print(best)
    a = rename_dir_by_formula(r'G:\high-throughput-workflow\332_opt')
