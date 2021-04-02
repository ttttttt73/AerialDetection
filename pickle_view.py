import pickle
import sys

if __name__=="__main__":
    argv = sys.argv
    if len(argv) <= 1:
        print("Specify pickle file as parameter")
    else:
        # print(pickle.load(open(argv[1], "rb")))
        pkl = pickle.load(open(argv[1], "rb"))
        for i in range(len(pkl[0])):
            print(pkl[0][i])
        print(len(pkl[0]))
