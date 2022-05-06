from map import Database
from modeling import Model, XOMOModel, POM3Model

model_db = Database(Model,
            {
                "xomo" : XOMOModel,
                "osp" : XOMOModel,
                "osp2" : XOMOModel,
                "ground" : XOMOModel,
                "flight" : XOMOModel,
                "pom3a" : POM3Model,
                "pom3b" : POM3Model,
                "pom3c" : POM3Model,
                "pom3d" : POM3Model,
            },
            {
                "xomo" : { },
                "osp" : { },
                "osp2" : { },
                "ground" : { },
                "flight" : { },
                "pom3a" : { },
                "pom3b" : { },
                "pom3c" : { },
                "pom3d" : { },
            }
    
)
