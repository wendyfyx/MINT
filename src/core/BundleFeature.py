from aenum import AutoNumberEnum

class BundleFeature(AutoNumberEnum, init='full_name', start=1):
    fa = 'Fractional Anisotropy'
    md = 'Mean Diffusivity'
    rd = 'Radial Diffusivity'
    axd = 'Axial Diffusivity'

    @property
    def filename(self):
        if self.name == 'AxD':
            return 'ad'
        else:
            return self.name.lower()

def feature_to_int(name):
    '''Convert bundle name to integer value'''
    return BundleFeature[name].value-1+3

def int_to_feature(value):
    '''Convert integer to bundle name'''
    return BundleFeature(value+1-3).name