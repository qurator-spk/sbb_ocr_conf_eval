import sys
import click
from ocrd_models.ocrd_page import parse

def extract_confs(page_fname):
    """
    Load PAGE-XML file, return lists of textline and word confidences
    """
    pcgts = parse(page_fname)

    confs_textline, confs_word = [], []

    for textline in pcgts.get_Page().get_AllTextLines():
        textline_conf = textline.get_TextEquiv()[0].conf
        if textline_conf is not None:
            confs_textline.append(textline_conf)
        for word in textline.get_Word():
            word_conf = word.get_TextEquiv()[0].conf
            if word_conf is not None:
                confs_word.append(word_conf)
    print(confs_textline, confs_word)
    # 
    # try:
    #     xmlns = str(root.tag).split("}")[0].strip("{")
    # except IndexError:
    #     xmlns = "No namespace found."
    # 
    # if "alto" in root.tag:
    #     wc_path = f".//{{{xmlns}}}String"
    #     wc_attr = "WC"
    # elif "PAGE" in root.tag:
    #     wc_path = f".//{{{xmlns}}}TextEquiv"
    #     wc_attr = "conf"
    # else:
    #     return 0, 0, 0, 0
    # 
    # confidences = []
    # for conf in xml.iterfind(wc_path):
    #         wc = float(conf.attrib.get(wc_attr))
    #         if wc is not None:
    #             confidences.append(wc)
    # 
    # if confidences:
    #     confidences_array = np.array(confidences)
    #     mean = round(np.mean(confidences_array), 3) 
    #     median = round(np.median(confidences_array), 3)
    #     variance = round(np.var(confidences_array), 3)
    #     standard_deviation = round(np.std(confidences_array), 3)
    #     
    #     return mean, median, variance, standard_deviation  
    # else:
    #     return 0, 0, 0, 0

if __name__ == '__main__':
    statistics(sys.argv[1])
