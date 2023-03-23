# Design-and-implementation-of-garbage-classifier
    近年来，我国在各大城市通过推行政策以及相关法律法规来推进垃圾分类。本文在现有的垃圾分类倡导之下，基于深度学习对生活垃圾类别的检测进行研究，并结合研究成果设计了生活垃圾分类识别系统，辅助人们完成生活垃圾分类，推动垃圾分类政策的落实。本文的主要研究内容如下:
    （1）本文针对深度学习中目标检测网络对于小目标的检测率低的问题，引入基于空间的自适应特征融合算法对YOLOv4网络结构进行改进，通过自适应特征融合对提取的特征在空间上进行过滤，缓解尺度不一致性问题，从而提升小目标的检测率。同时针对于单阶段目标检测算法训练过程中存在正负样本不平衡的问题，本文引入焦点损失函数对YOLOv4进行改进。最后，通过实验验证，本文采取的改进算法对于模型的准确率有着明显地提升。
    （2）本文面向生活垃圾分类的实际使用场景，对于生活垃圾分类的相关需求进行了细致地调研、分析和细化，并根据系统需求分析对系统进行详细设计与实现。结合本文的研究成果，设计并实现了生活垃圾分类识别系统，系统分为生活垃圾分类识别小程序模块以及后台管理系统模块，使用C/S 架构搭建生活垃圾分类识别小程序模块，使用B/S架构搭建后台管理系统模块，系统功能主要分为用户信息管理、新闻资讯、垃圾分类、知识竞赛、积分兑换、系统管理6个模块，并对系统功能进行了相关测试。
    关键词：垃圾分类，深度学习，目标检测，特征融合，损失函数

    In recent years, my country has promoted garbage classification by implementing policiesand relevant laws and regulations in major cities. Under the current advocacy of garbageclassification,this paper studies the detection of domestic garbage categories based on deeplearning, and designs a domestic garbage classification and identification system based on theresearch results to assist people in completing domestic garbage classification and promote theimplementation of garbage classification policies. The main research contents of this paper areas follows:
    (1) In view of the low detection rate of object detection network for small targets in deeplearning，a spatial-based adaptive feature fusion algorithm is introduced to improve theYOLOv4 network structure, and the extracted features are spatially processed through adaptivefeature fusion. Filter to alleviate the scale inconsistency problem，thereby improving thedetection rate of small targets.At the same time,in view of the imbalance of positive andnegative samples in the training process of the single-stage object detection algorithm，thispaper introduces the focal loss function to improve YOLOv4.Finally，through experimentalverification，the improved algorithm adopted in this paper has significantly improved theaccuracy of the model.
    (2) For the actual usage scenarios of domestic waste classification, this paper conducts adetailed investigation, analysis and refinement of the relevant requirements of domestic wasteclassification，and designs and implements the system in detail according to the analysis ofsystem requirements.Combined with the research results of this paper，a domestic wasteclassification and identification system is designed and implemented. The system is dividedinto a domestic waste classification and identification applet module and a backgroundmanagement system module. The C/S architecture is used to build a domestic wasteclassification and identification applet module, and the B/S The framework builds abackground management system module. The system functions are mainly divided into 6modules: user information management, news information, garbage classification, knowledgecompetition, points exchange, and system management. The system functions are tested.
    Keywords:waste classification,deep learning,object detection,feature fusion,loss function

    参考文献
    [1]蔡兆彬. 基于深度学习的生活垃圾分类识别系统的设计与实现[D].东北林业大学,2022.DOI:10.27009/d.cnki.gdblu.2022.001684.
    [2]陈伟. 基于深度学习的垃圾分类算法研究[D].天津职业技术师范大学,2021.DOI:10.27711/d.cnki.gtjgc.2021.000065.
    [3]董子源. 基于深度学习的垃圾分类系统设计与实现[D].中国科学院大学(中国科学院沈阳计算技术研究所),2020.DOI:10.27587/d.cnki.gksjs.2020.000005.
    [4]周鸿利. 基于深度学习的垃圾分类识别算法研究[D].华中科技大学,2020.DOI:10.27157/d.cnki.ghzku.2020.006408.
    [5]潘唯一. 基于深度学习的垃圾分类识别方法研究与实现[D].成都理工大学,2020.DOI:10.26986/d.cnki.gcdlc.2020.000336.
    [6]战秋成,季龙华,赵际云,修艳琪,戴婷婷.基于深度学习的智能垃圾分类系统研究[J].机械工程师,2022(08):100-103.
    [7]陈牧图,谭睿,石垒垒,冯月芹.基于深度学习的智能垃圾分类系统设计[J].电子测试,2022,36(17):12-14+18.DOI:10.16520/j.cnki.1000-8519.2022.17.003.
    [8]陶航,江学焕,张金亮,陈波.基于深度学习的垃圾分类系统[J].湖北汽车工业学院学报,2022,36(02):36-39+44.
    [9]李丕兵. 基于深度学习的垃圾分类系统研究与应用[D].青岛大学,2021.DOI:10.27262/d.cnki.gqdau.2021.002259.
    [10]徐丽. 基于深度学习的垃圾分类系统的设计与开发[D].浙江大学,2021.DOI:10.27461/d.cnki.gzjdx.2021.002525.
    [11]田震. 基于深度学习的垃圾分类算法研究[D].哈尔滨工业大学,2021.DOI:10.27061/d.cnki.ghgdu.2021.004427.
    [12]梁旭东. 基于深度学习的智能垃圾分类系统研究[D].西安建筑科技大学,2021.DOI:10.27393/d.cnki.gxazu.2021.001038.
    [13]莫卓亚,彭创权.基于深度学习的垃圾分类识别技术[J].现代工业经济和信息化,2020,10(10):60-61.DOI:10.16525/j.cnki.14-1362/n.2020.10.26.
    [14]武嘉年. 基于深度学习的生活垃圾分类检测及定位的研究[D].合肥学院,2022.DOI:10.27876/d.cnki.ghfxy.2022.000018.
    [15]王小燕,谢文昊,杨艺芳,胡瑞.基于深度学习的垃圾分类检测方法[J].现代电子技术,2021,44(21):110-113.DOI:10.16652/j.issn.1004-373x.2021.21.023.
