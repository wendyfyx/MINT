from aenum import AutoNumberEnum

class Atlas30(AutoNumberEnum, init='full_name', start=1):
    AF_L = 'Left Arcuate Fasciculus'
    AF_R = 'Right Arcuate Fasciculus'
    CC_ForcepsMajor = 'Corpus Callosum Major'
    CC_ForcepsMinor= 'Corpus Callosum Minor'
    CCMid = 'Corpus Callosum Mid'
    CST_L = 'Left Corticospinal Tract'
    CST_R = 'Right Corticospinal Tract'
    EMC_L = 'Left Extreme Capsule'
    EMC_R = 'Right Extrame Capsule'
    FPT_L = 'Left Frontopontine Tract'
    FPT_R = 'Right Frontopontine Tract'
    IFOF_L = 'Left Inferior Fronto-occipital Fasciculus'
    IFOF_R = 'Right Inferior Fronto-occipital Fasciculus'
    ILF_L = 'Left Inferior Longitudinal Fasciculus'
    ILF_R = 'Right Inferior Longitudinal Fasciculus'
    MdLF_L = 'Left Middle Longitudinal Fasciculus'
    MdLF_R = 'Right Middle Longitudinal Fasciculus'
    MLF_L = 'Left Medial Longitudinal fasciculus'
    MLF_R = 'Right Medial Longitudinal fasciculus'
    ML_L = 'Left Medial Lemniscus'
    ML_R = 'Right Medial Lemniscus'
    OPT_L = 'Left Occipito Pontine Tract'
    OPT_R = 'Right Occipito Pontine Tract'
    OR_L = 'Left Optic Radiation'
    OR_R = 'Right Optic Radiation'
    STT_L = 'Left Spinothalamic Tract'
    STT_R = 'Right Spinothalamic Tract'
    UF_L = 'Left Uncinate Fasciculus'
    UF_R = 'Right Uncinate Fasciculus'
    V = 'Vermis'

    @property
    def hemisphere(self):
        if 'left' in self.full_name.lower():
            return 'L'
        elif 'right' in self.full_name.lower():
            return 'R'
        else:
            return 'C'
        
    @property
    def type(self):
        projection = ['CST','F','FPT','OPT','OR']
        association = ['AF','C','EMC','IFOF','ILF','MdLF','SLF','UF']
        commissural = ['CC','CCMid']
        brainstem = ['ML','MLF','STT']
        cerebellum = ['CB','V']
        if self.name.split('_')[0] in projection:
            return 'Projection'
        elif self.name.split('_')[0] in association:
            return 'Association'
        elif self.name.split('_')[0] in commissural:
            return 'Commissural'
        elif self.name.split('_')[0] in brainstem:
            return 'Brainstem'
        elif self.name.split('_')[0] in cerebellum:
            return 'Cerebellum'
        else:
            return 'UNK'

def name_to_int(name):
    '''Convert bundle name to integer value'''
    return Atlas30[name].value-1

def int_to_name(value):
    '''Convert integer to bundle name'''
    return Atlas30(value+1).name