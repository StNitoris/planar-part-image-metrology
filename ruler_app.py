"""Desktop entry point for selecting ruler marks and the measurement region."""
import os
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2 as cv
import numpy as np

from ruler_measure import batch_summary, measure, read_image


ROOT = Path(__file__).resolve().parent
PROMPTS = ["点选第一条刻线", "点选第二条刻线的同一端",
           "点选零件选框的一个角，给板边留少量白纸", "点选选框对角，避开尺身",
           "检查刻线间距和零件选框，然后点击「测量」"]


class App:
    def __init__(self, root):
        self.root = root
        root.title("直尺标定 · 平面零件尺寸测量")
        root.geometry(f"{min(1160,root.winfo_screenwidth()-80)}x{min(850,root.winfo_screenheight()-100)}")
        top = ttk.Frame(root,padding=7)
        top.pack(fill="x")
        ttk.Button(top,text="选择照片",command=self.choose).pack(side="left")
        ttk.Label(top,text="  两条刻度间距 (mm) ").pack(side="left")
        self.distance = tk.StringVar(value="100")
        ttk.Entry(top,textvariable=self.distance,width=8).pack(side="left")
        for caption,action in [("撤回一点",self.undo),("适应窗口",self.fit),
                               ("测量",self.process),("接受并继续",self.accept),("打开结果",self.open_result)]:
            ttk.Button(top,text=caption,command=action).pack(side="left",padx=3)
        self.status = tk.StringVar(value="选择一张或多张带尺原图。滚轮放大，按住右键拖动。")
        ttk.Label(root,textvariable=self.status,padding=(9,5)).pack(fill="x")
        self.canvas = tk.Canvas(root,bg="#2b3036",highlightthickness=0)
        self.canvas.pack(fill="both",expand=True)
        self.canvas.bind("<Button-1>",self.click)
        self.canvas.bind("<MouseWheel>",self.zoom)
        self.canvas.bind("<ButtonPress-3>",lambda event: setattr(self,"drag",np.array([event.x,event.y])))
        self.canvas.bind("<B3-Motion>",self.pan)
        self.canvas.bind("<Configure>",lambda _: self.draw())
        self.image = None
        self.view = None
        self.origin = np.zeros(2)
        self.mag = 1.
        self.points = []
        self.rows = []
        self.paths = []
        self.current = None
        self.output = None
        self.preview = None
        self.pending = None

    def choose(self):
        paths = filedialog.askopenfilenames(title="选择同一零件的照片",filetypes=[
            ("图像","*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),("所有文件","*.*")])
        if not paths:
            return
        self.output = ROOT/"测量结果"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.rows = []
        self.paths = list(map(Path,paths))
        self.next_photo()

    def next_photo(self):
        if not self.paths:
            self.status.set(f"已完成 {len(self.rows)} 张，可点击「打开结果」查看汇总。")
            self.current = None
            return
        self.current = self.paths.pop(0)
        self.image = self.view = read_image(self.current)
        self.pending = None
        self.preview = None
        self.points = []
        self.fit()
        self.prompt()

    def prompt(self):
        self.status.set(f"{self.current.name}　{PROMPTS[len(self.points)]}。滚轮放大，右键拖动。")

    def fit(self):
        if self.view is None:
            return
        h,w = self.view.shape[:2]
        cw,ch = self.canvas.winfo_width(),self.canvas.winfo_height()
        self.mag = min(cw/w,ch/h)
        self.origin = (np.array([cw,ch])/[self.mag,self.mag]-[w,h])*(-.5)
        self.draw()

    def draw(self):
        if self.view is None:
            return
        cw,ch = self.canvas.winfo_width(),self.canvas.winfo_height()
        affine = np.array([[self.mag,0,-self.origin[0]*self.mag],[0,self.mag,-self.origin[1]*self.mag]])
        img = cv.warpAffine(self.view,affine,(cw,ch),borderValue=(54,48,43))
        if self.preview is None:
            pts = np.rint((np.asarray(self.points).reshape(-1,2)-self.origin)*self.mag).astype(int)
            if len(pts)>=2:
                cv.line(img,tuple(pts[0]),tuple(pts[1]),(0,135,255),1)
            if len(pts)==4:
                cv.rectangle(img,tuple(pts[2]),tuple(pts[3]),(255,180,0),2)
            for i,pt in enumerate(pts):
                cv.drawMarker(img,tuple(pt),(10,80,255),cv.MARKER_CROSS,16,2)
                cv.putText(img,str(i+1),tuple(pt+[5,-6]),cv.FONT_HERSHEY_SIMPLEX,.65,(30,60,255),2)
        self.bitmap = tk.PhotoImage(data=cv.imencode(".ppm",img)[1].tobytes(),format="PPM")
        self.canvas.delete("all")
        self.canvas.create_image(0,0,image=self.bitmap,anchor="nw")

    def click(self,event):
        if self.current is None or self.preview is not None or len(self.points)>=4:
            return
        pt = np.rint(self.origin+np.array([event.x,event.y])/self.mag).astype(int)
        if np.any(pt<0) or np.any(pt>=self.image.shape[1::-1]):
            return
        self.points.append(pt.tolist())
        self.draw()
        self.prompt()

    def zoom(self,event):
        if self.view is None:
            return
        anchor = np.array([event.x,event.y])
        new = np.clip(self.mag*(1.35 if event.delta>0 else 1/1.35),.04,8)
        self.origin += anchor*(1/self.mag-1/new)
        self.mag = new
        self.draw()

    def pan(self,event):
        now = np.array([event.x,event.y])
        self.origin += (self.drag-now)/self.mag
        self.drag = now
        self.draw()

    def undo(self):
        if self.current is None:
            return
        if self.preview is not None:
            self.view = self.image
            self.preview = self.pending = None
            self.fit()
        if self.points:
            self.points.pop()
        self.draw()
        self.prompt()

    def process(self):
        if self.current is None or self.preview is not None:
            return
        if len(self.points)!=4:
            self.prompt()
            return
        corners = np.asarray(self.points[2:])
        roi = [*corners.min(axis=0),*corners.max(axis=0)]
        run = self.output/(self.current.stem+"_"+datetime.now().strftime("%H%M%S_%f"))
        try:
            mm = float(self.distance.get())
            self.status.set("正在计算轮廓和孔口…")
            self.root.update_idletasks()
            data = measure(self.image,self.points[:2],mm,roi,run,self.current)
            result_img = read_image(run/"06_dimensions.png")
        except (ValueError,OSError,cv.error,np.linalg.LinAlgError) as exc:
            self.prompt()
            messagebox.showerror("测量未完成",str(exc))
            return
        self.preview = run
        self.pending = data
        self.view = result_img
        self.fit()
        self.status.set("核对板边和孔口后点「接受并继续」；需调整选框可点「撤回一点」。")

    def accept(self):
        if self.pending is None or self.current is None:
            return
        self.rows.append((self.current.name,self.pending))
        batch_summary(self.output,self.rows)
        self.pending = None
        self.next_photo()

    def open_result(self):
        folder = self.preview if self.pending is not None else self.output
        if folder is not None and folder.is_dir():
            os.startfile(folder)


if __name__=="__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
