import cv2

ref_point=[]
cropping=False
sel_rect_endpoint=[]
dragging=False
idx=-1
curr_points=[]
corners=[]
def get_coor_rect(image):
        global ref_point,cropping,sel_rect_endpoint
        img=image.copy()
        def click_and_crop(event,x,y,flags,param):
                global ref_point,cropping,sel_rect_endpoint
                if event==cv2.EVENT_LBUTTONDOWN:
                        ref_point=[(x,y)]
                        cropping=True
                elif event==cv2.EVENT_LBUTTONUP:
                        ref_point.append((x,y))
                        cropping=False
                        res=cv2.rectangle(img,ref_point[0],ref_point[1],(255,0,0),2)
                        cv2.imshow("image",img)
                elif event==cv2.EVENT_MOUSEMOVE and cropping:
                        sel_rect_endpoint.append((x,y))
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)
        while True:
                if not cropping:
                        cv2.imshow("image",img)
                elif cropping and sel_rect_endpoint:
                        rect_copy=img.copy()
                        res1=cv2.rectangle(rect_copy,sel_rect_endpoint[0],sel_rect_endpoint[-1],(0,0,255),2)
                        cv2.imshow("image",rect_copy)
                key=cv2.waitKey(1) & 0xFF
                if key==ord("r"):
                        img=image.copy()
                        sel_rect_endpoint=[]
                elif key==ord("c"):
                        break
        if len(ref_point)==2:
                [x1,y1]=list(ref_point[0])
                [x2,y2]=list(ref_point[1])
                roi=image[min(y1,y2):max(y1,y2),min(x1,x2):max(x1,x2)]
                cv2.imshow('ROI',roi)
        [x1,y1]=list(ref_point[0])
        [x2,y2]=list(ref_point[1])
        corners=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        return corners

def in_position(points,pt):
        (x,y)=pt
        for center in points:
                res=(x-center[0])**2 + (y-center[1])**2
                if res<=25:
                        return points.index(center)
        return -1

def draw_edges(Img,points,idx=-1):
        if idx!=-1:
                for pts in points:
                        if pts!=points[idx]:
                                res=cv2.circle(Img,pts,5,(255,0,0),-1)
                for i in range(len(points)):
                        if i!=idx and (i+1)%4!=idx:
                                res=cv2.line(Img,points[i],points[(i+1)%4],(255,0,0),2)
        else:
                for pts in points:
                        res=cv2.circle(Img,pts,5,(255,0,0),-1)
                for i in range(len(points)):
                        res=cv2.line(Img,points[i],points[(i+1)%4],(255,0,0),2)
        return Img

def adjust_coor_quad(image,rect):
        print("Drag the corners to the suitable positions...")
        print("Press c when done")
        print("Press r to reset")
        global dragging,corners,curr_points,idx
        img=image.copy()
        corners=[tuple(i) for i in rect]
        img1=draw_edges(img,corners)
        img=img1.copy()
        def drag_and_drop(event,x,y,flags,param):
                global dragging,corners,curr_points,idx
                if event==cv2.EVENT_LBUTTONDOWN:
                        idx=in_position(corners,(x,y))
                        if idx!=-1:
                                curr_points=[]
                                dragging=True
                        else:
                                dragging=False
                elif event==cv2.EVENT_LBUTTONUP and idx!=-1:
                        img=image.copy()
                        corners[idx]=(x,y)
                        dragging=False
                        img=draw_edges(img,corners)
                        cv2.imshow("image",img)
                elif event==cv2.EVENT_MOUSEMOVE and dragging:
                        curr_points.append((x,y))
                        cv2.setMouseCallback("image", drag_and_drop)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", drag_and_drop)
        while True:
                if not dragging:
                        cv2.imshow("image",img)
                elif dragging and curr_points:
                        img=image.copy()
                        img=draw_edges(img,corners,idx)
                        res=cv2.line(img,curr_points[-1],corners[(idx+1)%4],(255,0,0),2)
                        res=cv2.line(img,curr_points[-1],corners[(idx-1)%4],(255,0,0),2)
                        res=cv2.circle(img,curr_points[-1],5,(255,0,0),-1)
                        cv2.imshow("image",img)
                key=cv2.waitKey(1) & 0xFF
                if key==ord("r"):
                        img=img1.copy()
                        curr_points=[]
                        corners=[tuple(i) for i in rect]
                elif key==ord("c"):
                        break
        cv2.destroyAllWindows()
        return corners
