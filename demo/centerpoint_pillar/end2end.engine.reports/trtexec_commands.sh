# build
trtexec --onnx=demo/centerpoint_pillar/end2end.onnx --minShapes=coors:10000x4\,voxels:10000x20x5\,num_points:10000 --optShapes=coors:40000x4\,voxels:40000x20x5\,num_points:40000 --maxShapes=coors:50000x4\,voxels:50000x20x5\,num_points:50000 --saveEngine=demo/centerpoint_pillar/end2end.engine --fp16 --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=demo/centerpoint_pillar/end2end.engine.reports/build_layer_info.json --dumpProfile --exportProfile=demo/centerpoint_pillar/end2end.engine.reports/build_profile.json 

# profile
trtexec --loadEngine=demo/centerpoint_pillar/end2end.engine --shapes=coors:40000x4\,voxels:40000x20x5\,num_points:40000 --warmUp=100 --duration=10 --iterations=100 --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=demo/centerpoint_pillar/end2end.engine.reports/profile_layer_info.json --dumpProfile --exportProfile=demo/centerpoint_pillar/end2end.engine.reports/profile_profile.json 

