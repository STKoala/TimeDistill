# TimeDistill

使用的数据集 monash 实际是 LightGTS项目所用处理后的版本

- 手动蒸馏训练 chronos-2 ：

使用方式:

```python
python chronos2_distill_gkd.py

```

`chronos2_distill.py` 是基于 tlr 实现的，还没试过。

`chronos2_distill_gkd.py` 目前是没用到该框架的特色，学生的生成只能用自己，而是依旧是学生可以看过去的。但用到了 JSD 散度