import sys,re
from nltk.corpus import stopwords

STOPWORDS = "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their" \
            " theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the" \
            " and but if or because as until while of at by for with about against between into through during before after above below to from up down in" \
            " out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only" \
            " own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn" \
            " needn shan shouldn wasn weren won wouldn".strip().split()

STOPWORDS.extend(list(stopwords.words('english')))
STOPWORDS = set(STOPWORDS)
P_DATA = './persona_with_role.txt'

def readPersona_role(file,output):
    """
        read from file, write lines describing persona to output
        :param file: original file to read from, eg. train_both_original.txt
        :param output: output file
        :return:
    """
    personas = []
    num_of_conversation = 0
    persona_tmp = []
    current_role = ""
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.split()
            if line[0] == '1':
                num_of_conversation += 1
                current_role = "new"
            if line[2] == "persona:":
                role = line[1]
                if (role != current_role) & (current_role != ""):
                    personas.append("   ".join(persona_tmp) + '\n')
                    persona_tmp = []
                persona_tmp.append(" ".join(line[3:]))
                current_role = role

    with open(output,'w') as op:
        op.writelines(personas)


def get_stats(rm_stopword,file=P_DATA):
    """
        calculate vocabulary size of input file
        :param rm_stopword: set true will remove stopwords from the corpus
        :return:
    """

    content = []
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if rm_stopword:
                content.extend(text_clean_w2vv(line))
            else:
                content.extend(line.split(" "))
    unique_words = set(content)
    print("Vocab size is: {}".format(len(unique_words)))



def text_clean_w2vv(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    cleaned_string = string.strip().lower().split()
    return [word for word in cleaned_string if word not in STOPWORDS]

if __name__ == "__main__":
    # readPersona_role("train_both_original.txt","persona_with_role.txt")

    get_stats(True)
