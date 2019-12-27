 https://zhuanlan.zhihu.com/p/27624517 



深度使用过tensorflow的同学可能都使用过tensorboard，因为tensorboard这一高级的可视化的工具，很多人对tensorflow也爱不释手，目前除了tensorflow之外还没有哪个深度学习库开发出了一套完美的可视化工具，这也是tensorflow流行的原因之一，如果不使用tensorboard，你想可视化训练流程，那么你只能自己保存变量，自己画曲线。

所以有很多使用别的深度学习框架的人在研究如何将tensorboard移植到他们的框架中来，当然也有很多成功的例子，不然我也不会写这篇文章了，下面我们就来讲几种目前流行的方法。

1.使用Crayon

Crayon是一个支持任何语言使用tensorboard的框架，它的说明文档访问下面的[网址](https://link.zhihu.com/?target=https%3A//github.com/torrvision/crayon)，目前他只支持Python和Lua，而且安装过程比较麻烦，需要docker，不推荐使用此方法。

2.使用tensorboard_logger

tensorboard_logger是由TeamHG-Memex开发的使用tensorboard的库，可以访问[文档界面](https://link.zhihu.com/?target=https%3A//github.com/TeamHG-Memex/tensorboard_logger)，安装也略微有点繁琐，需要安装tensorflow和他们开发的tensorboard_logger，安装完成之后按照文档的使用说明就可以使用tensorboard了。

3.导入一个脚本实现tensorboard

这个办法是我认为最简单的办法，也是我目前使用的办法，只需要安装cpu版的tensorflow，通过pip install tensorflow就能够很快安装上，然后只需要复制这个[网址](https://link.zhihu.com/?target=https%3A//github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py)里面的代码到你的项目文件目录，新建一个logger.py的文件，将代码复制进去就ok了。

然后在你的python文件里面输入from logger import Logger，然后在训练之前定义好想存放tensorboard文件的文件夹，logger = Logger('./logs')这里可以使用任何文件夹存放tensorboard文件。

然后在训练过程中可以通过下面的方式记录想要记录的变量

```python
# (1) Log the scalar values
info = {
    'loss': loss.data[0],
    'accuracy': accuracy.data[0]
}

for tag, value in info.items():
    logger.scalar_summary(tag, value, step)

# (2) Log values and gradients of the parameters (histogram)
for tag, value in model.named_parameters():
    tag = tag.replace('.', '/')
    logger.histo_summary(tag, to_np(value), step)
    logger.histo_summary(tag+'/grad', to_np(value.grad), step)

# (3) Log the images
info = {
    'images': to_np(img.view(-1, 28, 28)[:10])
}

for tag, images in info.items():
    logger.image_summary(tag, images, step)
```

这样我们就将我们需要的变量放进了tensorborad中，然后我们在当前目录下输入tensorbard --logdir='./logs'，这里需要输入自己的文件夹名称，我的文件夹之前定义为了logs，然后你就能够看到下面的界面

![img](https://pic4.zhimg.com/80/v2-db7e0c44b49460b9468e50bf9ef5ae37_hd.png)



在浏览器中输入[http://0.0.0.0:6006/](https://link.zhihu.com/?target=http%3A//0.0.0.0%3A6006/)，你就能够进到tensorboard界面了，就像下面这样

![img](https://pic3.zhimg.com/80/v2-3603082c6b5dfffd942220350cd7bf22_hd.png)

![img](https://pic2.zhimg.com/80/v2-70c78e92845da9b1bbcdc6d9a518d741_hd.png)

![img](https://pic2.zhimg.com/80/v2-07a63f1285a486d8512b3ba8635a3b35_hd.png)



这样我们就能够成功地在PyTorch中使用tensorboard可视化了，是不是很方便呢。

本文参考自[yunjey’s github](https://link.zhihu.com/?target=https%3A//github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)

* 完整代码已经上传到了[github](https://link.zhihu.com/?target=https%3A//github.com/SherlockLiao/pytorch-beginner/tree/master/04-Convolutional%20Neural%20Network)上
  * 注意使用 `tensorflow 1.10`版本

欢迎查看我的知乎专栏，[深度炼丹](https://zhuanlan.zhihu.com/c_94953554)

欢迎访问我的[博客](https://link.zhihu.com/?target=https%3A//sherlockliao.github.io/)