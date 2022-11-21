"""
Copyright {2018} {Viraj Mavani}
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
"""

import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image

# import miscellaneous modules
import os
import numpy as np
import tensorflow as tf
import config
import tf_config
import custom_config
import math
from pascal_voc_writer import Writer
import sys

# make sure the file is inside semi-auto-image-annotation-tool-master
import pathlib
# cur_path = pathlib.Path(__file__).parent.absolute()
cur_path = pathlib.Path(__file__).parent.absolute().as_posix()
sys.path.append(cur_path)
os.chdir(cur_path)

# these should be equal for image to not appear stretched, but if not should not influence results
VIEW_SCALE_X = 2
VIEW_SCALE_Y = 2

DEFAULT_MODEL_PATH = "/home/adi/Projects/traffic_lights/traffic_lights_detection/"
DEFAULT_IMGS_DIR_PATH = "/home/adi/bags/2022-09-27_traffic_lights_data/2022-09-27-10-58-39_sat10/oak/frames"
MERGE_NEARBY = True

IMG_RES_DESIRED = (640, 480)
IMG_CROP = (0.2, 0, 0.6, 1.0)


class MainGUI:
    def __init__(self, master):

        # to choose between keras or tensorflow models
        self.model_type = "custom"  # default
        self.models_dir = ''  # gets updated as per user choice
        self.model_path = ''

        sys.path.insert(1, DEFAULT_MODEL_PATH)
        from pipeline_yolov3_autoware_classifier import PipelineYoloV3AutowareClassifier
        from data_loaders import LoadImages
        self.model_pipeline = PipelineYoloV3AutowareClassifier(debug=False)
        self.data_loader = LoadImages(source_path="", crop=IMG_CROP, resize=IMG_RES_DESIRED)

        self.img_w_desired = IMG_RES_DESIRED[0] * (IMG_CROP[3] - IMG_CROP[1])
        self.img_h_desired = IMG_RES_DESIRED[1] * (IMG_CROP[2] - IMG_CROP[0])
        print(f"desired image size: {self.img_w_desired} {self.img_h_desired}")

        self.parent = master
        self.parent.title("Semi Automatic Image Annotation Tool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=False, height=False)

        # Initialize class variables
        self.image_path = None
        self.image_name = None
        self.img = None
        self.tkimg = None
        self.imageDir = ''
        self.imageDirPathBuffer = ''
        self.imageList = []
        self.imageTotal = 0
        self.imageCur = 0
        self.cur = 0
        self.bboxIdList = []
        self.labelsList = []
        self.bboxList = []
        self.bboxPointList = []
        self.o1 = None
        self.o2 = None
        self.o3 = None
        self.o4 = None
        self.bboxId = None
        self.currLabel = None
        self.editbboxId = None
        self.currBboxColor = None
        self.zoomImgId = None
        self.zoomImg = None
        self.zoomImgCrop = None
        self.tkZoomImg = None
        self.hl = None
        self.vl = None
        self.editPointId = None
        self.filename = None
        self.filenameBuffer = None
        self.objectLabelList = []
        self.EDIT = False
        self.autoSuggest = StringVar()
        self.autoSuggest.set(str(1))
        self.writer = None
        self.thresh = 0.0
        self.org_h = 0
        self.org_w = 0
        # initialize mouse state
        self.STATE = {'x': 0, 'y': 0}
        self.STATE_COCO = {'click': 0}

        # initialize annotation file
        self.anno_filename = 'annotations.csv'
        self.annotation_file = open('annotations/' + self.anno_filename, 'w+')
        self.annotation_file.write("")
        self.annotation_file.close()

        # ------------------ GUI ---------------------

        # Control Panel
        self.ctrlPanel = Frame(self.frame)
        self.ctrlPanel.grid(row=0, column=0, sticky=W + N)
        self.openBtn = Button(self.ctrlPanel, text='Open', command=self.open_image)
        self.openBtn.grid(columnspan=2, sticky=W + E)
        self.openDirBtn = Button(self.ctrlPanel, text='Open Dir', command=self.open_image_dir)
        self.openDirBtn.grid(columnspan=2, sticky = W + E)

        self.nextBtn = Button(self.ctrlPanel, text='Next -->', command=self.open_next)
        self.nextBtn.grid(columnspan=2, sticky=W + E)
        self.previousBtn = Button(self.ctrlPanel, text='<-- Previous', command=self.open_previous)
        self.previousBtn.grid(columnspan=2, sticky=W + E)
        self.saveBtn = Button(self.ctrlPanel, text='Save', command=self.save)
        self.saveBtn.grid(columnspan=2, sticky=W + E)
        self.autoManualLabel = Label(self.ctrlPanel, text="Suggestion Mode")
        self.autoManualLabel.grid(columnspan=2, sticky=W + E)
        self.radioBtnAuto = Radiobutton(self.ctrlPanel, text="Auto", variable=self.autoSuggest, value=1)
        self.radioBtnAuto.grid(row=7, column=0, sticky=W + E)
        self.radioBtnManual = Radiobutton(self.ctrlPanel, text="Manual", variable=self.autoSuggest, value=2)
        self.radioBtnManual.grid(row=7, column=1, sticky=W + E)
        self.semiAutoBtn = Button(self.ctrlPanel, text="Detect", command=self.automate)
        self.semiAutoBtn.grid(columnspan=2, sticky=W + E)
        self.disp = Label(self.ctrlPanel, text='Coordinates:')
        self.disp.grid(columnspan=2, sticky=W + E)

        self.mb = Menubutton(self.ctrlPanel, text="COCO Classes for Suggestions", relief=RAISED)
        self.mb.grid(columnspan=2, sticky=W + E)
        self.mb.menu = Menu(self.mb, tearoff=0)
        self.mb["menu"] = self.mb.menu

        self.addCocoBtn = Button(self.ctrlPanel, text="+", command=self.add_labels_coco)
        self.addCocoBtn.grid(columnspan=2, sticky=W + E)
        self.addCocoBtnAllClasses = Button(self.ctrlPanel, text="Add All Classes", command=self.add_all_classes)
        self.addCocoBtnAllClasses.grid(columnspan=2, sticky=W + E)

        # options to add different models
        self.mb1 = Menubutton(self.ctrlPanel, text="Select models from here", relief=RAISED)
        self.mb1.grid(columnspan=2, sticky=W + E)
        self.mb1.menu = Menu(self.mb1, tearoff=0)
        self.mb1["menu"] = self.mb1.menu

        self.addModelBtn = Button(self.ctrlPanel, text="Add model", command=self.add_model)
        self.addModelBtn.grid(columnspan=2, sticky=W + E)

        self.zoomPanelLabel = Label(self.ctrlPanel, text="Precision View Panel")
        self.zoomPanelLabel.grid(columnspan=2, sticky=W + E)
        self.zoomcanvas = Canvas(self.ctrlPanel, width=150, height=150)
        self.zoomcanvas.grid(columnspan=2, sticky=W + E)

        # Image Editing Region
        self.canvas = Canvas(self.frame, width=int(self.img_w_desired * VIEW_SCALE_X), height=int(self.img_h_desired * VIEW_SCALE_Y))
        self.canvas.grid(row=0, column=1, sticky=W + N)
        self.canvas.bind("<Button-1>", self.mouse_click)
        self.canvas.bind("<Motion>", self.mouse_move, "+")
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        self.parent.bind("<Key-Left>", self.open_previous)
        self.parent.bind("<Key-Right>", self.open_next)
        self.parent.bind("Escape", self.cancel_bbox)

        # Labels and Bounding Box Lists Panel
        self.listPanel = Frame(self.frame)
        self.listPanel.grid(row=0, column=2, sticky=W + N)
        self.listBoxNameLabel = Label(self.listPanel, text="List of Objects").pack(fill=X, side=TOP)
        self.objectListBox = Listbox(self.listPanel, width=40)
        self.objectListBox.pack(fill=X, side=TOP)
        self.delObjectBtn = Button(self.listPanel, text="Delete", command=self.del_bbox)
        self.delObjectBtn.pack(fill=X, side=TOP)
        self.clearAllBtn = Button(self.listPanel, text="Clear All", command=self.clear_bbox)
        self.clearAllBtn.pack(fill=X, side=TOP)
        self.classesNameLabel = Label(self.listPanel, text="Classes").pack(fill=X, side=TOP)
        self.textBox = Entry(self.listPanel, text="Enter label")
        self.textBox.pack(fill=X, side=TOP)
        self.addLabelBtn = Button(self.listPanel, text="+", command=self.add_label).pack(fill=X, side=TOP)
        self.delLabelBtn = Button(self.listPanel, text="-", command=self.del_label).pack(fill=X, side=TOP)

        self.labelListBox = Listbox(self.listPanel)
        self.labelListBox.pack(fill=X, side=TOP)

        self.addThresh = Label(self.listPanel, text="Threshold").pack(fill=X, side=TOP)
        self.textBoxTh = Entry(self.listPanel, text="Enter threshold value")
        self.textBoxTh.pack(fill=X, side=TOP)
        self.enterthresh = Button(self.listPanel, text="Set", command=self.changeThresh).pack(fill=X, side=TOP)

        if self.model_type == "keras":
            self.cocoLabels = config.labels_to_names.values()
        elif self.model_type == "tensorflow":
            self.cocoLabels = tf_config.labels_to_names.values()
        else:
            self.cocoLabels = custom_config.labels_to_names.values()

        self.cocoIntVars = []
        for idxcoco, label_coco in enumerate(self.cocoLabels):
            self.cocoIntVars.append(IntVar())
            self.mb.menu.add_checkbutton(label=label_coco, variable=self.cocoIntVars[idxcoco])
        # print(self.cocoIntVars)

        self.modelIntVars = []
        for idxmodel, modelname in enumerate(self.available_models()):
            self.modelIntVars.append(IntVar())
            self.mb1.menu.add_checkbutton(label=modelname, variable=self.modelIntVars[idxmodel])

        # STATUS BAR
        self.statusBar = Frame(self.frame, width=self.img_w_desired)
        self.statusBar.grid(row=1, column=1, sticky=W + N)
        self.processingLabel = Label(self.statusBar, text="                      ")
        self.processingLabel.pack(side="left", fill=X)
        self.imageIdxLabel = Label(self.statusBar, text="                      ")
        self.imageIdxLabel.pack(side="right", fill=X)

        # load all classes by default
        self.add_all_classes()

        # open default image dir
        self.open_image_dir_from_path(DEFAULT_IMGS_DIR_PATH)

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def available_models(self):
        self.models_dir = os.path.join(cur_path, 'snapshots')
        # only for keras and tf
        model_categ = [dir_ for dir_ in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, dir_))]
        # creating all model options list
        model_names = []
        for categ in model_categ:
            for name in os.listdir(os.path.join(self.models_dir , categ)):
                model_names.append(os.path.join(categ,name))
        return model_names


    def changeThresh(self):
        if(float(self.textBoxTh.get()) >0 and float(self.textBoxTh.get()) <1):
            self.thresh = float(self.textBoxTh.get())

    def open_image(self):
        self.filename = filedialog.askopenfilename(title="Select Image", filetypes=(("jpeg files", "*.jpg"),
                                                                                    ("all files", "*.*")))
        if not self.filename:
            return None
        self.filenameBuffer = self.filename
        self.load_image(self.filenameBuffer)

    def open_image_dir(self):
        path = filedialog.askdirectory(title="Select Dataset Directory")
        self.open_image_dir_from_path(path)

    def open_image_dir_from_path(self, path):
        self.imageDir = path
        if not self.imageDir:
            return None
        print(self.imageDir)
        self.imageList = os.listdir(self.imageDir)
        self.imageList = sorted(self.imageList)
        self.imageTotal = len(self.imageList)
        self.filename = None
        self.imageDirPathBuffer = self.imageDir
        print(self.imageTotal)
        print(self.cur)
        self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])

    def open_video_file(self):
        pass

    def load_image(self, file):
        self.img = Image.open(file)
        self.image_path = file
        self.image_name, _ = os.path.splitext(os.path.basename(self.image_path))
        self.imageCur = self.cur + 1
        self.imageIdxLabel.config(text='  ||   Image: %s - Number: %d / %d' % (self.image_name, self.imageCur, self.imageTotal))
        # Resize to Pascal VOC format
        w, h = self.img.size
        self.org_w, self.org_h = self.img.size
        if self.model_type != "custom":
            if w >= h:
                baseW = 640
                wpercent = (baseW / float(w))
                hsize = int((float(h) * float(wpercent)))
                self.img = self.img.resize((baseW, hsize), Image.BICUBIC)
            else:
                baseH = 480
                wpercent = (baseH / float(h))
                wsize = int((float(w) * float(wpercent)))
                self.img = self.img.resize((wsize, baseH), Image.BICUBIC)
            self.img_cv = np.array(self.img)
        else:
            open_cv_image = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
            # use data_loader process method. img should be BGR before. returns img in RGB
            self.img_cv, _ = self.data_loader.process(open_cv_image)
            self.img = Image.fromarray(self.img_cv)     # convert back to PIL

        # scale img view
        self.scaled_view_img = self.img.resize((int(self.img.size[0] * VIEW_SCALE_X), int(self.img.size[1] * VIEW_SCALE_Y)), Image.BICUBIC)
        self.tkimg = ImageTk.PhotoImage(self.scaled_view_img)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.clear_bbox()
        print(f"-------------- {self.image_name} ------------------")

    def open_next(self, event=None):
        self.save()
        if self.cur < len(self.imageList):
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()
        if self.autoSuggest.get() == str(1):
            self.automate()

    def open_previous(self, event=None):
        self.save()
        if self.cur > 0:
            self.cur -= 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()
        if self.autoSuggest.get() == str(1):
            self.automate()

    def save(self):
        # save annotations to annotations.csv and VOC format
        if self.filenameBuffer is None:
            w, h = self.img.size
            self.writer = Writer(os.path.join(self.imageDirPathBuffer , self.imageList[self.cur]), w, h)
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                x1, y1, x2, y2 = self.bboxList[idx]
                self.writer.addObject(str(self.objectLabelList[idx]), x1, y1, x2, y2)
                self.annotation_file.write(self.imageDirPathBuffer + '/' + self.imageList[self.cur] + ',' +
                                           ','.join(map(str, self.bboxList[idx])) + ',' + str(self.objectLabelList[idx])
                                           + '\n')
            self.annotation_file.close()
            baseName = os.path.splitext(self.imageList[self.cur])[0]
            save_dir = 'annotations/annotations_voc/'
            save_path = save_dir + baseName + '.xml'
            if(not os.path.exists(save_dir)):
                os.mkdir(save_dir)

            self.writer.save(save_path)
            self.writer = None
        else:
            w, h = self.img.size
            self.writer = Writer(self.filenameBuffer, w, h)
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                x1, y1, x2, y2 = self.bboxList[idx]
                self.writer.addObject(str(self.objectLabelList[idx]), x1, y1, x2, y2)
                self.annotation_file.write(self.filenameBuffer + ',' + ','.join(map(str, self.bboxList[idx])) + ','
                                           + str(self.objectLabelList[idx]) + '\n')
            self.annotation_file.close()
            baseName = os.path.splitext(self.imageList[self.cur])[0]
            self.writer.save('annotations/annotations_voc/' + baseName + '.xml')
            self.writer = None

        # TODO save in YOLO format
        # One row per object
        # Each row is class x_center y_center width height format.
        # Box coordinates must be in normalized xywh format (from 0 - 1).
        # If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
        # Class numbers are zero-indexed (start from 0).

        img_w, img_h = self.img.size
        if self.bboxList:
            results_file_path = self.imageDir + "/" + self.image_name + ".txt"
            with open(results_file_path, 'w') as write_file:
                for idx, box in enumerate(self.bboxList):
                    xywh = self.xyxy_to_xywhnorm(box, img_w, img_h)
                    label_idx = [key for key, value in custom_config.labels_to_names.items() if value == self.objectLabelList[idx]]     # TODO fix this
                    write_file.write(f"{label_idx[0]} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")

            print(f"saved to file: {results_file_path}")

    # convert bounding box from [min_x, min_y, max_x, max_y] to normalized [center_x, center_y, w, h]
    def xyxy_to_xywhnorm(self, box_xyxy, img_width, img_height):
        x_center = ((box_xyxy[2] + box_xyxy[0]) / 2) / img_width
        y_center = ((box_xyxy[3] + box_xyxy[1]) / 2) / img_height
        width = (box_xyxy[2] - box_xyxy[0]) / img_width
        height = (box_xyxy[3] - box_xyxy[1]) / img_height

        return (x_center, y_center, width, height)

    def mouse_click(self, event):
        # Check if Updating BBox
        if self.canvas.find_enclosed(event.x - 10, event.y - 10, event.x + 10, event.y + 10):
            self.EDIT = True
            self.editPointId = int(self.canvas.find_enclosed(event.x - 10, event.y - 10, event.x + 10, event.y + 10)[0])
        else:
            self.EDIT = False

        # Set the initial point
        if self.EDIT:
            idx = self.bboxPointList.index(self.editPointId)
            self.editbboxId = self.bboxIdList[math.floor(idx/4.0)]
            self.bboxId = self.editbboxId
            pidx = self.bboxIdList.index(self.editbboxId)
            pidx = pidx * 4
            self.o1 = self.bboxPointList[pidx]
            self.o2 = self.bboxPointList[pidx + 1]
            self.o3 = self.bboxPointList[pidx + 2]
            self.o4 = self.bboxPointList[pidx + 3]
            if self.editPointId == self.o1:
                a, b, c, d = self.canvas.coords(self.o3)
            elif self.editPointId == self.o2:
                a, b, c, d = self.canvas.coords(self.o4)
            elif self.editPointId == self.o3:
                a, b, c, d = self.canvas.coords(self.o1)
            elif self.editPointId == self.o4:
                a, b, c, d = self.canvas.coords(self.o2)
            self.STATE['x'], self.STATE['y'] = int((a+c)/2), int((b+d)/2)
        else:
            self.STATE['x'], self.STATE['y'] = event.x, event.y

    def mouse_drag(self, event):
        self.mouse_move(event)
        if self.bboxId:
            self.currBboxColor = self.canvas.itemcget(self.bboxId, "outline")
            self.canvas.delete(self.bboxId)
            self.canvas.delete(self.o1)
            self.canvas.delete(self.o2)
            self.canvas.delete(self.o3)
            self.canvas.delete(self.o4)
        if self.EDIT:
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=self.currBboxColor)
        else:
            self.currBboxColor = config.COLORS[len(self.bboxList) % len(config.COLORS)]
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=self.currBboxColor)

    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        self.zoom_view(event)
        if self.tkimg:
            # Horizontal and Vertical Line for precision
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
            # elif (event.x, event.y) in self.bboxBRPointList:
            #     pass

    def mouse_release(self, event):
        try:
            labelidx = self.labelListBox.curselection()
            self.currLabel = self.labelListBox.get(labelidx)
        except:
            pass
        if self.EDIT:
            self.update_bbox()
            self.EDIT = False
        x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
        y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
        scaled_x1 = int(x1 / VIEW_SCALE_X)
        scaled_x2 = int(x2 / VIEW_SCALE_X)
        scaled_y1 = int(y1 / VIEW_SCALE_Y)
        scaled_y2 = int(y2 / VIEW_SCALE_Y)

        self.bboxList.append((scaled_x1, scaled_y1, scaled_x2, scaled_y2))
        o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
        o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
        o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
        o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")
        self.bboxPointList.append(o1)
        self.bboxPointList.append(o2)
        self.bboxPointList.append(o3)
        self.bboxPointList.append(o4)
        self.bboxIdList.append(self.bboxId)
        l = self.canvas.create_text(x1 , y1 - 10, fill=self.currBboxColor, text=str(self.currLabel))
        self.labelsList.append(l)
        self.bboxId = None
        self.objectLabelList.append(str(self.currLabel))
        self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (scaled_x1, scaled_y1, scaled_x2, scaled_y2) + ': ' + str(self.currLabel))
        self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                      fg=self.currBboxColor)
        self.currLabel = None

    def zoom_view(self, event):
        try:
            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            self.zoomImgCrop = self.zoomImg.crop(((event.x - 25), (event.y - 25), (event.x + 25), (event.y + 25)))
            self.zoomImgCrop = self.zoomImgCrop.resize((150, 150))
            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
            hl = self.zoomcanvas.create_line(0, 75, 150, 75, width=2)
            vl = self.zoomcanvas.create_line(75, 0, 75, 150, width=2)
        except:
            pass

    def update_bbox(self):
        idx = self.bboxIdList.index(self.editbboxId)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectListBox.delete(idx)

        # if a label is selected, change box label
        try:
            labelidx = self.labelListBox.curselection()
            self.currLabel = self.labelListBox.get(labelidx)
            print(f"set label to {self.currLabel}")
        except:
            print("no curr label -> keep old label")
            self.currLabel = self.objectLabelList[idx]

        self.objectLabelList.pop(idx)
        idx_pts = idx*4
        self.canvas.delete(self.bboxPointList[idx_pts])
        self.canvas.delete(self.bboxPointList[idx_pts+1])
        self.canvas.delete(self.bboxPointList[idx_pts+2])
        self.canvas.delete(self.bboxPointList[idx_pts+3])
        self.bboxPointList.pop(idx_pts)
        self.bboxPointList.pop(idx_pts)
        self.bboxPointList.pop(idx_pts)
        self.bboxPointList.pop(idx_pts)
        self.canvas.delete(self.labelsList[idx])
        self.labelsList.pop(idx)

    def cancel_bbox(self, event):
        if self.STATE['click'] == 1:
            if self.bboxId:
                self.canvas.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def del_bbox(self):
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.canvas.delete(self.bboxIdList[idx])
        self.canvas.delete(self.bboxPointList[idx * 4])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 1])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 2])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 3])
        self.canvas.delete(self.labelsList[idx])
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxIdList.pop(idx)
        self.labelsList.pop(idx)
        self.bboxList.pop(idx)
        self.objectLabelList.pop(idx)
        self.objectListBox.delete(idx)

    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        for idx in range(len(self.bboxPointList)):
            self.canvas.delete(self.bboxPointList[idx])
        for idx in range(len(self.labelsList)):
            self.canvas.delete(self.labelsList[idx])
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []
        self.bboxPointList = []
        self.labelsList.clear()

    def add_label(self):
        if self.textBox.get() != '':
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if self.textBox.get() not in curr_label_list:
                self.labelListBox.insert(END, str(self.textBox.get()))
            self.textBox.delete(0, 'end')

    def del_label(self):
        labelidx = self.labelListBox.curselection()
        self.labelListBox.delete(labelidx)

    def add_model(self):
        for listidxmodel, list_model_name in enumerate(self.available_models()):
            if (self.modelIntVars[listidxmodel].get()):
                # check which model is it keras or tensorflow
                self.model_path = os.path.join(self.models_dir, list_model_name)
                # if its Tensorflow model then modify path
                if ('keras' in list_model_name):
                    self.model_type = "keras"
                elif ('tensorflow' in list_model_name):
                    self.model_path = os.path.join(self.model_path, 'frozen_inference_graph.pb')
                    self.model_type = "tensorflow"
                    # change cocoLabels corresponding to tensorflow
                    self.cocoLabels = tf_config.labels_to_names.values()
                elif 'torch' in list_model_name:
                    self.model_type = "torch"
                    self.model_path = os.path.join(self.models_dir, list_model_name)
                break

    def add_labels_coco(self):
        for listidxcoco, list_label_coco in enumerate(self.cocoLabels):
            if self.cocoIntVars[listidxcoco].get():
                curr_label_list = self.labelListBox.get(0, END)
                curr_label_list = list(curr_label_list)
                if list_label_coco not in curr_label_list:
                    self.labelListBox.insert(END, str(list_label_coco))

    def add_all_classes(self):
        for listidxcoco, list_label_coco in enumerate(self.cocoLabels):
            # if self.cocoIntVars[listidxcoco].get():
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if list_label_coco not in curr_label_list:
                self.labelListBox.insert(END, str(list_label_coco))

    def automate(self):
        self.processingLabel.config(text="Processing     ")
        self.processingLabel.update_idletasks()
        if self.model_type == "custom":
            # execute pipeline
            valid_detections, invalid_detections = self.model_pipeline.execute(self.img_cv)

            if not (valid_detections + invalid_detections):
                print("No detected objects")
                self.processingLabel.config(text="None detected       ")
                return

            if MERGE_NEARBY:
                related_detections = self.model_pipeline.detect_tls_relationship(valid_detections)

            detections = valid_detections + invalid_detections
            m_name = "YoloV3+Classifier"
            boxes = list()
            labels = list()
            scores = list()
            masks = list()
            for det in detections:
                print(det)

                boxes.append(det.bounding_box)  #xyxy
                pipeline_label = f"{det.color}-{det.shape}"
                # convert pipeline labels to new labels
                label = custom_config.pipeline_labels_new_labels[f"{det.color}"][f"{det.shape}"]
                labels.append(label)
                # score only used for threshold filtering here, but not needed since done inside the pipeline
                scores.append(det.class_confidence)

            boxes = np.expand_dims(np.asarray(boxes), axis=0)
            labels = np.expand_dims(np.asarray(labels), axis=0)
            scores = np.expand_dims(np.asarray(scores), axis=0)
            masks = np.expand_dims(np.asarray(masks), axis=0)

            config_labels = custom_config.labels_to_names

        # if tensorflow
        elif self.model_type == "tensorflow":
            # Convert RGB to BGR
            opencvImage = self.img_cv[:, :, ::-1].copy()
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                sess = tf.Session(graph=detection_graph)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image_expanded = np.expand_dims(opencvImage, axis=0)
            (boxes, scores, labels, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            config_labels = tf_config.labels_to_names
            m_name = os.path.split((os.path.split(self.model_path)[0]))[1]

        elif self.model_type == "keras":
            # Convert RGB to BGR
            opencvImage = self.img_cv[:, :, ::-1].copy()
            keras.backend.tensorflow_backend.set_session(self.get_session())
            model_path = self.model_path
            model = models.load_model(model_path, backbone_name='resnet50')
            image = preprocess_image(opencvImage)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            config_labels = config.labels_to_names
            m_name = os.path.split(self.model_path)[1]
        else:
            print("unknown model")
            return

        for idx, (box, label, score) in enumerate(zip(boxes[0], labels[0], scores[0])):
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if score < self.thresh:
                print(f"discarded detction with score {score}")
                continue

            label_str = config_labels[label]

            if label_str not in curr_label_list:
                print(f"label {label_str} not in curr_label_list")
                continue

            b = box
            # only if using tf models as keras and tensorflow have different coordinate order
            if (self.model_type == "tensorflow"):
                w, h = self.img.size
                (b[0],b[1],b[2],b[3]) = (b[1]*w, b[0]*h, b[3]*w, b[2]*h)
            b = b.astype(int)

            color = config.COLORS[len(self.bboxList) % len(config.COLORS)]
            scale_view_box = np.array(((int(b[0] * VIEW_SCALE_X), int(b[1] * VIEW_SCALE_Y),
                                        int(b[2] * VIEW_SCALE_X), int(b[3] * VIEW_SCALE_Y))))
            self.bboxId = self.canvas.create_rectangle(scale_view_box[0], scale_view_box[1],
                                                       scale_view_box[2], scale_view_box[3],
                                                       width=2,
                                                       outline=color)
            self.bboxList.append((b[0], b[1], b[2], b[3]))
            o1 = self.canvas.create_oval(scale_view_box[0] - 3, scale_view_box[1] - 3, scale_view_box[0] + 3, scale_view_box[1] + 3, fill="red")
            o2 = self.canvas.create_oval(scale_view_box[2] - 3, scale_view_box[1] - 3, scale_view_box[2] + 3, scale_view_box[1] + 3, fill="red")
            o3 = self.canvas.create_oval(scale_view_box[2] - 3, scale_view_box[3] - 3, scale_view_box[2] + 3, scale_view_box[3] + 3, fill="red")
            o4 = self.canvas.create_oval(scale_view_box[0] - 3, scale_view_box[3] - 3, scale_view_box[0] + 3, scale_view_box[3] + 3, fill="red")
            self.bboxPointList.append(o1)
            self.bboxPointList.append(o2)
            self.bboxPointList.append(o3)
            self.bboxPointList.append(o4)
            self.bboxIdList.append(self.bboxId)
            l = self.canvas.create_text(scale_view_box[0], scale_view_box[1] - 10, fill=color, text=label_str)
            self.labelsList.append(l)
            self.bboxId = None
            self.objectLabelList.append(str(label_str))
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (b[0], b[1], b[2], b[3]) + ': ' +
                                  str(label_str)+' '+str(int(score*100))+'%'
                                      +' '+ m_name)
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                          fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])
        self.processingLabel.config(text="Done              ")


if __name__ == '__main__':
    root = Tk()
    imgicon = PhotoImage(file='icon.gif')
    root.tk.call('wm', 'iconphoto', root._w, imgicon)
    tool = MainGUI(root)
    root.mainloop()
