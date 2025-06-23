import pickle as pkl


# with open("./datasets/mosi.pkl", 'rb') as f:
with open("E:/SoftwareandDataSet/MSA/DMD/MOSI/aligned_50.pkl", 'rb') as f:
    mosi = pkl.load(f)
test_mosi = mosi['test']
# (words, visual, acoustic), label, segment =  mosi['test'][0]

with open("./datasets/mosi.pkl", 'rb') as f:
    mosi_easy = pkl.load(f)
test_mosi_easy = mosi_easy['test']
# (words, visual, acoustic), label, segment =  mosi['test'][0]

print("WORDS", words)
print("VISUAL", visual.shape)
print("ACOUSTIC", acoustic.shape)
print("LABEL", label)
print("SEGMENT", segment)