TriBSC: Gene-anchored multimodal biological integration with complementary network evidence for miRNA-disease association prediction

PyTorch 2.8.0 
Python 3.12(ubuntu22.04) 
CUDA 12.8
numpy==2.4.6
scipy==1.18.1
scikit-learn==1.9.0
matplotlib==3.11.1

三个目录各自的主要功能如下：
D14_UnifiedDataCache：整合miRNA-疾病关联矩阵、实体名称及miRNA-gene、disease-gene、GO、Pathway、PPI等生物关系数据。模型主要使用其中的关联矩阵作为预测任务标签，并利用统一的实体顺序保证不同数据源之间正确对齐。
Bio_Foundation_Cache：保存miRNA序列和疾病文本的预训练特征。miRNA特征由RNA序列模型生成，疾病特征由生物医学语言模型生成，同时保存实体映射信息。此外，d6_gene_bridge_cache.npz 提供miRNA与疾病在基因层面的重叠特征。
D16_BioHeteroPath_Cache：在 miRNA-gene、disease-gene、GO、Pathway和PPI等关系基础上构建基因功能上下文，生成gene_context以及miRNA、疾病对应的基因索引、有效掩码和关系权重。模型利用这些数据建模miRNA与疾病之间潜在的基因介导生物学联系。