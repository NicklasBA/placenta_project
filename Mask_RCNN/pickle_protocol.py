import pickle

annotations = pickle.load(open("/home/s183993/placenta_project/Mask_RCNN/mask_rcnn.pkl"), 'rb')
with open('/home/s183993/placenta_project/Mask_RCNN/mask_rcnn.pkl', 'wb') as handle:
    pickle.dump(annotations, handle, protocol=4)