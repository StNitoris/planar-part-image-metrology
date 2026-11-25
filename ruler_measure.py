"""Two-point ruler calibration and planar plate measurement. Coordinates are original pixels."""
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def read_image(path):
    image = cv.imdecode(np.fromfile(path, np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取照片，请使用 JPG、PNG 或 BMP。")
    return image


def save(path, image):
    ok, data = cv.imencode(".png", image)
    if not ok:
        raise OSError(f"无法保存 {path}")
    data.tofile(path)


def fit_side(points):
    vx, vy, x, y = cv.fitLine(points.astype(np.float32), cv.DIST_HUBER, 0, .01, .01).ravel()
    return np.array([x, y]), np.array([vx, vy])


def intersection(a, b):
    p, u = a
    q, v = b
    t = np.linalg.solve(np.column_stack((u, -v)), q-p)[0]
    return p+t*u


def board_geometry(edge):
    contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("框内未发现轮廓。")
    contour = max(contours, key=cv.contourArea)
    if cv.contourArea(contour) < .28*edge.size:
        raise ValueError("板外轮廓没有闭合，请缩紧选框、避开尺子或调整反光。")
    points = contour.reshape(-1, 2)
    rect = cv.boxPoints(cv.minAreaRect(contour))
    sides = np.roll(rect, -1, axis=0)-rect
    u = sides[np.argmax(np.linalg.norm(sides, axis=1))]
    u = u/np.linalg.norm(u)
    if u[0] < 0:
        u = -u
    v = np.array([u[1], -u[0]])
    frame = (points-rect.mean(axis=0)) @ np.column_stack((u,v))
    lo, hi = frame.min(axis=0), frame.max(axis=0)
    span = hi-lo
    lines = []
    for axis, limit in [(0,lo[0]),(0,hi[0]),(1,lo[1]),(1,hi[1])]:
        other = 1-axis
        use = ((abs(frame[:,axis]-limit)<.04*span[axis]) &
               (frame[:,other]>lo[other]+.15*span[other]) &
               (frame[:,other]<hi[other]-.15*span[other]))
        if use.sum()<12:
            raise ValueError("直边取点不足，请检查选区是否包含完整零件。")
        lines.append(fit_side(points[use]))
    left,right,bottom,top = lines
    corners = np.array([intersection(left,bottom),intersection(right,bottom),
                        intersection(right,top),intersection(left,top)])
    return contour, corners


def fit_circle(points):
    origin = points.mean(axis=0)
    p = points-origin
    x,y,k = np.linalg.lstsq(np.column_stack((2*p,np.ones(len(p)))), (p*p).sum(axis=1), rcond=None)[0]
    return origin+[x,y], np.sqrt(max(0,k+x*x+y*y))


def hole_at_corner(image, corner, midpoint, short_side):
    # Restrict the circle search to this corner so brushed lines elsewhere cannot vote.
    side = max(30,int(.24*short_side))
    direction = (midpoint-corner)/np.linalg.norm(midpoint-corner)
    seed = corner+direction*(side*.52)
    x,y = np.maximum(0, np.floor(seed-side*.62).astype(int))
    end = np.minimum(image.shape[1::-1], np.ceil(seed+side*.62).astype(int))
    patch = image[y:end[1],x:end[0]]
    if min(patch.shape[:2])<20:
        return None
    smooth = cv.GaussianBlur(patch,(5,5),1.2)
    gray = cv.cvtColor(smooth, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(smooth,18,55)
    sat = cv.cvtColor(smooth,cv.COLOR_BGR2HSV)[:,:,1]
    _,paper = cv.threshold(sat,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    candidates = []
    for mask in (edges,paper):
        contours,_ = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            rough = np.sqrt(area/np.pi)
            if not short_side*.012 < rough < short_side*.075:
                continue
            circularity = 4*np.pi*area / max(1,cv.arcLength(cnt,True)**2)
            if circularity>.72:
                center,radius = fit_circle(cnt.reshape(-1,2))
                candidates.append((center,radius))
    if not candidates:
        circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1.2,side*.15,param1=80,param2=25,
                                  minRadius=max(4,int(short_side*.012)),maxRadius=max(8,int(short_side*.075)))
        if circles is not None:
            candidates.extend((c[:2],c[2]) for c in circles.reshape(-1,3))
    for center,rough in candidates:
        # Starting in the paper core, find the first strong local colour gradient
        # towards the metal. This avoids fitting the outer highlight of the lip.
        angles = np.linspace(0,2*np.pi,96,endpoint=False)
        directions = np.column_stack((np.cos(angles),np.sin(angles)))
        distances = np.arange(0,rough*1.38,.5)
        loc = (center+directions[:,None,:]*distances[None,:,None]).astype(np.float32)
        if loc.min()<0 or loc[:,:,0].max()>=gray.shape[1]-1 or loc[:,:,1].max()>=gray.shape[0]-1:
            continue
        vals = cv.remap(smooth,loc[:,:,0],loc[:,:,1],cv.INTER_LINEAR).astype(float)
        core = vals[:,distances<rough*.4]
        reference = np.median(core.reshape(-1,3),axis=0)
        delta = np.max(abs(vals-reference),axis=2)
        if np.percentile(delta[:,distances<rough*.4],95)>25:
            continue
        gradient = np.max(abs(vals[:,4:]-vals[:,:-4]),axis=2)
        in_ring = (distances[2:-2]>.72*rough) & (distances[2:-2]<1.26*rough)
        cutoff = np.maximum(3,gradient[:,in_ring].max(axis=1)*.28)
        peak = (gradient>=np.roll(gradient,1,axis=1)) & (gradient>np.roll(gradient,-1,axis=1))
        changed = in_ring & peak & (gradient>=cutoff[:,None])
        first = changed.argmax(axis=1)+2
        good = changed.any(axis=1)
        if good.sum()<76:
            continue
        points = center+directions[good]*distances[first[good],None]
        c,r = fit_circle(points)
        residual = abs(np.linalg.norm(points-c,axis=1)-r)
        keep = residual<max(2,.08*r)
        if keep.sum()<72 or abs(rough-r)>.35*rough:
            continue
        c,r = fit_circle(points[keep])
        return c+[x,y],r,points[keep]+[x,y]
    return None


def line_text(img, text, xy, color=(36,39,45), scale=.73):
    cv.putText(img,text,xy,cv.FONT_HERSHEY_SIMPLEX,scale,color,2,cv.LINE_AA)


def measure(image, ticks, distance_mm, roi, folder, source):
    ticks = np.asarray(ticks,float)
    separation = np.linalg.norm(ticks[1]-ticks[0])
    if separation<10 or not np.isfinite(distance_mm) or distance_mm<=0:
        raise ValueError("请在两条不同的刻线上选点，并输入正数毫米值。")
    scale = distance_mm/separation
    x,y,x2,y2 = map(int,roi)
    if not (0<=x<x2<=image.shape[1] and 0<=y<y2<=image.shape[0]) or min(x2-x,y2-y)<80:
        raise ValueError("选框太小或超出了照片范围，请重新框住整块零件。")
    folder.mkdir(parents=True,exist_ok=False)
    calib = image.copy()
    marks = ticks.round().astype(int)
    cv.line(calib,tuple(marks[0]),tuple(marks[1]),(0,150,255),3,cv.LINE_AA)
    for mark in marks:
        cv.drawMarker(calib,tuple(mark),(0,60,255),cv.MARKER_CROSS,35,3)
    cv.rectangle(calib,(x,y),(x2-1,y2-1),(240,110,0),3)
    line_text(calib,f"{distance_mm:g} mm / {separation:.1f} px = {scale:.6f} mm/px",(25,55),scale=1.1)
    save(folder/"01_scale.png",calib)
    crop = image[y:y2,x:x2]
    blur = cv.GaussianBlur(crop,(7,7),1.8)
    save(folder/"02_smoothed.png",blur)
    edge = cv.Canny(blur,22,65)
    # Bridge short gaps without filling entire highlight bands.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    closed = cv.morphologyEx(edge,cv.MORPH_CLOSE,kernel)
    save(folder/"03_edges.png",closed)
    contour,corners = board_geometry(closed)
    overlay = crop.copy()
    cv.drawContours(overlay,[contour],-1,(0,210,255),2)
    cv.polylines(overlay,[corners.round().astype(np.int32)],True,(255,170,20),3,cv.LINE_AA)
    bl,br,tr,tl = corners
    u = (br-bl)/np.linalg.norm(br-bl)
    v = np.array([u[1],-u[0]])
    length = ((br-bl)@u+(tr-tl)@u)/2
    width = ((tl-bl)@v+(tr-br)@v)/2
    result = {"L":float(length*scale),"W":float(width*scale)}
    table = [f"Length {length*scale:.2f} mm     Width {width*scale:.2f} mm",
             "Origin: lower left. Hole centres and visible aperture diameters (mm)."]
    detail = []
    for i,corner in enumerate(corners,1):
        hole = hole_at_corner(crop,corner,corners.mean(axis=0),width)
        keys = [f"H{i}_{suffix}" for suffix in ("X","Y","D")]
        if hole is None:
            result.update(dict.fromkeys(keys))
            table.append(f"H{i}   Review hole edge")
            continue
        c,r,points = hole
        values = [float((c-bl)@u*scale),float((c-bl)@v*scale),float(2*r*scale)]
        result.update(dict(zip(keys,values)))
        table.append(f"H{i}    X {values[0]:.2f}     Y {values[1]:.2f}     D {values[2]:.2f}")
        p = tuple(c.round().astype(int))
        cv.circle(overlay,p,int(round(r)),(255,60,200),3,cv.LINE_AA)
        cv.drawMarker(overlay,p,(255,60,200),cv.MARKER_CROSS,17,2)
        label_x = p[0]+int(r+4) if i in (1,4) else p[0]-int(r+65)
        line_text(overlay,f"H{i}",(label_x,p[1]),(255,60,200),1)
        rad = int(np.ceil(r*1.5))
        xx,yy = max(0,p[0]-rad),max(0,p[1]-rad)
        cut = crop[yy:p[1]+rad+1,xx:p[0]+rad+1].copy()
        for pt in points:
            cv.circle(cut,tuple(np.rint(pt-[xx,yy]).astype(int)),1,(255,30,180),-1)
        cv.circle(cut,(p[0]-xx,p[1]-yy),int(round(r)),(255,30,180),1,cv.LINE_AA)
        cut = cv.resize(cut,(250,250))
        line_text(cut,f"H{i}",(6,26),(255,80,190))
        detail.append(cut)
    save(folder/"04_fit.png",overlay)
    if detail:
        save(folder/"05_holes.png",np.hstack(detail))
    pic = cv.resize(overlay,(1150,round(1150*overlay.shape[0]/overlay.shape[1])),interpolation=cv.INTER_AREA)
    footer = np.full((len(table)*34+20,1150,3),249,np.uint8)
    for row,t in enumerate(table):
        line_text(footer,t,(16,30+34*row))
    save(folder/"06_dimensions.png",np.vstack((pic,footer)))
    record = dict(source=str(source),ticks_px=ticks.tolist(),distance_mm=distance_mm,
                  mm_per_px=float(scale),roi_xyxy=list(map(int,roi)),dimensions_mm=result)
    (folder/"measurement.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def batch_summary(folder, rows):
    keys = list(rows[0][1])
    with (folder/"measurements_mm.csv").open("w",newline="",encoding="utf-8-sig") as f:
        table = csv.writer(f)
        table.writerow(["image",*keys])
        for name,res in rows:
            table.writerow([name,*[res[k] for k in keys]])
    panel = np.full((105+36*len(keys),730,3),249,np.uint8)
    line_text(panel,"Selected photos (mm)",(20,33))
    for x,t in zip([20,170,275,475],["Feature","Count","Mean","Sample SD"]):
        line_text(panel,t,(x,72))
    for i,k in enumerate(keys):
        values = [res[k] for _,res in rows if res[k] is not None]
        cols = [k,str(len(values)),f"{np.mean(values):.2f}" if values else "--",
                f"{np.std(values,ddof=1):.3f}" if len(values)>1 else "--"]
        for x,t in zip([20,170,275,475],cols):
            line_text(panel,t,(x,108+i*36))
    save(folder/"summary.png",panel)
