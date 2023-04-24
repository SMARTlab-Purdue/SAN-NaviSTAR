class Static_Obstacles_State(object):
    def __init__(self, px, py,humans):
        self.px = px
        self.py = py
        self.humans = humans
        self.walls = [(-6,-6,-6,6),(-2,3,-2,6),(0,-6,0,-1),(2,3,2,6),(6,-6,6,6),(-6,6,6,6),(-6,3,-3,3),(-6,-6,6,-6),(-2,3,1,3),(2,3,5,3),(-6,-1,-1,-1),(0,-1,5,-1)]
        self.wall_lines,self.wall_k,self.wall_b = self.wall_line_equation()
        self.laser_lines,self.laser_k,self.laser_b = self.get_laser_line()
    def get_laser_line(self):
        laser_lines = []
        laser_k = []
        laser_b = []
        for i in range(6):
            line,k,b = self.laser_line_equation(i * 30)
            laser_lines.append(line)
            laser_k.append(k)
            laser_b.append(b)
        return laser_lines,laser_k,laser_b
    def judge_in_wall(self,point_x,point_y,j):
        if point_x >= self.walls[j][0] and point_x <= self.walls[j][2] and point_y >= self.walls[j][1] and point_y <= self.walls[j][3]:
            return True
        else:
            return False
    def laser_point_compute(self):
        left_laser_point = []
        right_laser_point = []
        cross_point = self.cross_point_compute()
        for point_list in cross_point:
            left_point = (-10, -10)
            right_point = (10, 10)
            for point in point_list:
                if point[0]-self.px < 0:
                    if point[0] > left_point[0]:
                        left_point = point
                elif point[0]-self.px > 0:
                    if point[0] < right_point[0]:
                        right_point = point
                elif point[1]-self.py < 0:
                    if point[1] > left_point[1]:
                        left_point = point
                elif point[1]-self.py > 0:
                    if point[1] < right_point[1]:
                        right_point = point
            if np.linalg.norm(left_point) > 5:
                left_point = (0,0)
            if np.linalg.norm(right_point) >5:
                right_point = (0,0)
            left_laser_point.append(left_point)
            right_laser_point.append(right_point)
        laser_point = right_laser_point[:4]+left_laser_point[4:6]+left_laser_point[:4]+right_laser_point[4:6]
        return laser_point
    def human_cross_point(self,theta):
        point = []
        if theta != 90:
            k = np.tan(theta * np.pi/180)
            b = self.py-k*self.px
            for human in self.humans:
                if np.around(np.abs(k*human.px-human.py+b)/np.sqrt(k**2+1),2)<human.radius:
                    point1,point2 = self.cross_point_line_circle(k,b,human.px,human.py,human.radius)
                    if point1 != None:
                        point.append(point1)
                        point.append(point2)
        else:
            for human in self.humans:
                if np.abs(human.px-self.px)<human.radius:
                    y1 = human.py + np.sqrt(human.radius**2-(self.px-human.px)**2)
                    y2 = human.py - np.sqrt(human.radius ** 2 - (self.px - human.px) ** 2)
                    point.append((self.px,y1))
                    point.append((self.px, y2))
        return point
    def cross_point_line_circle(self,k,b_l,x0,y0,r):
        a = 1+k**2
        b = 2*k*(b_l-y0)-2*x0
        c = x0**2+(b_l-y0)**2-r**2
        x1,x2 = self.get_root(a,b,c)
        if x1 != None:
            y1 = k*x1+b_l
            y2 = k*x2+b_l
            return (x1,y1),(x2,y2)
        else:
            return None,None
    def get_root(self,a,b,c):
        delta = b**2-4*a*c
        if delta >= 0:
            return (np.sqrt(delta)-b)/(2*a),(-np.sqrt(delta)-b)/(2*a)
        else:
            return None,None
    def cross_point_compute(self):
        cross_point = [[] for i in range(6)]
        for i in range(6):
            cross_point[i] = self.human_cross_point(i*30)
            if self.laser_k[i] == None:
                for j in range(len(self.wall_k)):
                    if self.wall_k[j] != None:
                        point_x = self.laser_lines[i]
                        point_y = self.wall_lines[j](self.laser_lines[i])
                        if self.judge_in_wall(point_x,point_y,j):
                            cross_point[i].append((point_x,point_y))
                    else:
                        if self.wall_lines[j] == self.laser_lines[i]:
                            if self.py < self.walls[j][1]:
                                cross_point[i].append((self.walls[j][0],self.walls[j][1]))
                            else:
                                cross_point[i].append((self.walls[j][2], self.walls[j][3]))
            else:
                for j in range(len(self.wall_k)):
                    if self.wall_k[j] == None:
                        point_x = self.wall_lines[j]
                        point_y = self.laser_lines[i](self.wall_lines[j])
                        if self.judge_in_wall(point_x, point_y, j):
                            cross_point[i].append((point_x, point_y))
                    else:
                        if self.laser_k[i] != self.wall_k[j]:
                            point_x = (self.wall_b[j]-self.laser_b[i])/(self.laser_k[i]-self.wall_k[j])
                            point_y = self.wall_lines[j](point_x)
                            if self.judge_in_wall(point_x, point_y, j):
                                cross_point[i].append((point_x, point_y))
                        else:
                            if self.laser_b[i] == self.wall_b[j]:
                                if np.abs(self.px-self.walls[j][0]) < np.abs(self.px-self.walls[j][2]):
                                    cross_point[i].append((self.walls[j][0], self.walls[j][1]))
                                else:
                                    cross_point[i].append((self.walls[j][2], self.walls[j][3]))
        return cross_point
    def laser_line_equation(self,theta):
        if theta != 90:
            k = np.tan(theta * np.pi/180)
            return lambda x: k * (x - self.px) + self.py,k,self.py-k*self.px
        else:
            return self.px,None,None
    def line_equation(self,k,x0,y0):
        return lambda x: k*(x - x0) + y0
    def wall_line_equation(self):
        wall_lines = []
        k_line = []
        b_line = []
        for wall in self.walls:
            if wall[2] - wall[0] != 0:
                k = (wall[3] - wall[1])/(wall[2] - wall[0])
                k_line.append(k)
                wall_lines.append(self.line_equation(k,wall[0],wall[1]))
                b_line.append(wall[1]-k*wall[0])
            else:
                wall_lines.append(wall[0])
                k_line.append(None)
                b_line.append(None)
        return wall_lines,k_line,b_line


    def __str__(self):
        return ' '.join([str(x) for x in self.laser_point_compute()])

import numpy as np
class Human():
    def __init__(self, px, py,radius):
        self.px = px
        self.py = py
        self.radius = radius
humans = []
human = Human(2,0,1)
humans.append(human)
c = Static_Obstacles_State(0, 0,humans)
p = c.laser_point_compute()
print(p)
print(len(p))