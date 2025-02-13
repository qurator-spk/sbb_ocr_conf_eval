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
