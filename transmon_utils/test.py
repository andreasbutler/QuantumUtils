from transmon import Transmon

if __name__=="__main__":
    print('hi')
    qubit_1 = Transmon.transmon_from_frequency_parameterization(5, -0.25)
    print(qubit_1.josephson_energy)
