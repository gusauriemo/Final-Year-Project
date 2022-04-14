import pandas as pd
import matplotlib.pyplot as plt

base_test = "test loss excel path"
base_training = "training loss excel path"

def no_decoder_fun(file_path, runtime, error_type):
        reference = pd.read_excel(file_path)
        reference = reference[reference.runtime == runtime]
        reference.reset_index(drop=True, inplace=True)
        reference.index +=1
        for i in range(2,11):
            reference.at[i, error_type] = reference.iloc[0][error_type]

        reference[error_type].plot(label= "No Decoder")
        plt.legend()

def training_fun(error_type):
    """error_type is in the format "DTE", that is, it is a string. Possible values are: DTE, DL, LDL, CL, LCL."""
    plt.figure()
    for lr in ["2e-4", "2e-5", "5e-4", "8e-5"]:

        df2 = pd.read_excel(base_training+"/"+lr+'.xlsx')
        df2.dropna(inplace=True)
        df2 = df2[df2.runtime == 190]
        df2.reset_index(drop=True, inplace=True)
        df2.index +=1
        df2[error_type].plot(title= "Training Accuracy", label= lr)
        plt.ylabel("Accuracy under "+error_type+" (%)")
        plt.xlabel("Epoch")
        plt.xlim(1,10)
        plt.legend()
    no_decoder_fun(base_training+"/no_decoder.xlsx", 190, error_type)
    plt.grid()
    plt.show()

def test_fun(error_type):
    """error_type is in the format "DTE", that is, it is a string. Possible values are: DTE, DL, LDL, CL, LCL."""
    plt.figure()

    for lr in ["2e-4", "2e-5", "5e-4", "8e-5"]:
        df1= pd.read_excel(base_test+"/"+lr+'.xlsx')
        df1.dropna(inplace=True)
        df1 = df1[df1.runtime == 57]
        df1.reset_index(drop=True, inplace=True)
        df1.index +=1
        df1[error_type].plot(title= "Test Accuracy", label= lr)
        plt.ylabel("Accuracy under "+error_type+" (%)")
        plt.xlabel("Epoch")
        plt.xlim(1,10)
        plt.legend()

    no_decoder_fun(base_test+"/no_decoder.xlsx", 57, error_type)
    plt.grid()
    plt.show()

for i in ["DTE", "DL", "LDL", "CL", "LCL"]:
    training_fun(i)
    test_fun(i)