 
from roboflow import Roboflow
rf = Roboflow(api_key="OogxLzQUmp71dRsom49O")
project = rf.workspace("jbnu-fi5vn").project("drone3_latest")
version = project.version(2)
dataset = version.download("yolov12")
                
