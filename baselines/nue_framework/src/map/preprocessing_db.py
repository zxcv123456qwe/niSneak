from map import Database
from preprocessing import PreProcessor
from preprocessing import FairSmoteSelector, DummyProcessor, WFO

preprocessing_db = Database(PreProcessor,
    {
        "none" : DummyProcessor,
        "fairsmote" : FairSmoteSelector,
        "wfo" : WFO,
    },
    {
        "fairsmote" : {
            "secondary" : True,
        },
        "wfo" : {
            "features" : None,
            "r0" : 0.0,
            "r1" : 0.3,
            "ri" : 0.03
        }
    }
)