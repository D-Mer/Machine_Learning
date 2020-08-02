这是我对这源码加了自己注释的版本，方便后续学习

本代码源自[https://github.com/sunshineatnoon/LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer)

关于原论文的解读可以查看我的博客[论文解读报告](https://blog.csdn.net/weixin_43959709/article/details/106847669)

原论文链接[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_Linear_Transformations_for_Fast_Image_and_Video_Style_Transfer_CVPR_2019_paper.pdf)





/Artistic/：存放生成的结果图

/data/：

---/content/：存放内容图

---/style/：存放风格图

/libs/：存放主要功能代码

---/Criterion.py：计算loss的方法

---/Loader.py：数据集加载类

---/Matrix.py：进行风格迁移的主要代码

---/models.py：包含网络模型

---/其余不是很重要

/models/：存放训练好的模型，因为模型比较大，所以我只留了链接

/salience_img/：存放生成的结果图和相应的显著性图

/gpu_test.py：用来测试gpu是否可用

/saliency.py：我们写的计算显著性的方法，采用FT算法

/TestArtistic.py：运行风格迁移代码的入口

/Train.py：训练模型用的代码，因为没有设备所以没有实际跑过



