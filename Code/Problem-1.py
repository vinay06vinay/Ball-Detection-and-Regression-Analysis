import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
def compute_landing_spot(x_center,y_center,unknowns):
    y_landing_spot = y_center[0] + 300
    '''To calculate landing spot of x given y
    we use the parabola equation  derived and solve for roots of equation.
    '''
    a = unknowns[0]
    b = unknowns[1]
    c = unknowns[2] - y_landing_spot
    dis = b * b - 4 * a * c 
    sqrt_val = math.sqrt(abs(dis)) 
    # checking condition for discriminant
    if dis > 0: 
        solution1 = (-b + sqrt_val)/(2 * a)
        solution2 = (-b - sqrt_val)/(2 * a)
    if(solution1 >0):
        print("The landing x-coordinate of the ball :",int((solution1)))
    else:
        print("The landing x-coordinate of the ball :",int((solution2)))
    return True
    
def plot_raw(x_center,y_center):
    fig,ax=plt.subplots()
    #The alternate ball coordinates are mappped for plot to appear nice
    ax.scatter(x_center[0:-1:3],y_center[0:-1:3])
    plt.title("Trajectory of Ball center")
    ax.set_xlabel('Width of Pixels in Image')
    ax.set_ylabel('Height of Pixels in Image')
    plt.gca().invert_yaxis()
    plt.show()
    return True
def compute_parabola(x_center,y_center):
    '''
    1. A parabole is fitted using the centers derived from ball trajectory
    '''
    x = np.array(x_center)
    y = np.array(y_center)
    x_sum = np.sum(x)
    x_2_sum = np.sum(np.power(x,2))
    x_3_sum = np.sum(np.power(x,3))
    x_4_sum = np.sum(np.power(x,4))
    y_sum = np.sum(y)
    xy_sum = np.sum(x*y)
    x_2_y_sum = np.sum(np.power(x,2)*y)
    a= np.array([[x_4_sum,x_3_sum,x_2_sum],[x_3_sum,x_2_sum,x_sum],[x_2_sum,x_sum,len(x_center)]])
    b= np.array([x_2_y_sum,xy_sum,y_sum])
    unknowns = np.linalg.solve(a,b)
    y_final =  (unknowns[0]*(x**2))+(unknowns[1]*x)+unknowns[2]
    print(f"The equation of parabola derived using curve fitting is y = {round(unknowns[0],3)}*(x**2) {round(unknowns[1],3)}*x + {round(unknowns[2],3)}")
    fig,ax=plt.subplots()
    plt.gca().invert_yaxis()
    plt.title("Parabola curving fitting using ball center coordinates")
    ax.set_xlabel('Width of Pixels in Image')
    ax.set_ylabel('Height of Pixels in Image')
    plt.plot(x,y_final,'r',label = 'Parabolic curve')
    ax.scatter(x_center,y_center,s=5,label = 'raw data')
    ax.legend()
    plt.show()
    return (unknowns)

def main():
    '''
    1. First using video capture, each frame of the video is read and image processing is done
    2. As part of image processing, the red ball in the image is masked using inrange and bitwise functions.
    3. The center of the ball is calculated using the mean of x and y coordinates of pixels which have value of 255 
    4. The frame is displayed with center on the ball
    '''
    
    video_object = cv2.VideoCapture("ball.mov")
    x_center = []
    y_center = []
    if (video_object.isOpened == False):
        print("Error Streaming the video")
    count = 0
    while (video_object.isOpened):
        ret, frame = video_object.read()
        if ret == True:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            low_range = np.array([0,175,110],np.uint8)
            high_range = np.array([4,255,255],np.uint8)
            mask= cv2.inRange(hsv,low_range,high_range)
            res = cv2.bitwise_and(frame,frame, mask= mask)
            blur = cv2.GaussianBlur(res, (5, 5),cv2.BORDER_DEFAULT)
            d=np.where(mask!=0)
            if(len(d[0]) >0):
                x = np.nanmean(d[1])
                y = np.nanmean(d[0])
                # Calculation of center by taking a diagonal 
                # x= (max(d[1]) + min(d[1]))/2
                # y= (max(d[0]) + min(d[0]))/2
                x_center.append(x)
                y_center.append(y)
                cv2.circle(frame,(int(x),int(y)),1,(255,255,0),2)
            cv2.imshow("Frame",frame)
            if(count == 100):
                masked_image = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                a = np.hstack([res,masked_image,hsv])
                cv2.imwrite("stack.jpg",a)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        count +=1 
    
    video_object.release()
    cv2.destroyAllWindows()
    plot_raw(x_center,y_center)
    unknowns = compute_parabola(x_center,y_center)
    time.sleep(2)
    compute_landing_spot(x_center,y_center,unknowns)
    
if __name__ == '__main__':
    main()
