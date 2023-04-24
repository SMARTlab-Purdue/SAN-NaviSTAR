import numpy as np


class Distance_to_Wall():
    def __init__(self,px,py):
        self.wall_v = np.array([[(-6,-6),(-6,6)],[(-2,3),(-2,6)],[(0,-6),(0,-1)],[(2,3),(2,6)],[(6,-6),(6,6)]])
        self.wall_h = np.array([[(-6, -6), (6, -6)], [(-6, -1), (6, -1)], [(-6, 3), (6, 3)], [(-6, 6), (6, 6)]])
        self.px = px
        self.py = py
        self.room = [0 for i in range(6)]
        self.room[0] = [self.wall_v[0]] +[self.wall_v[1]] + [self.wall_h[2]] +[self.wall_h[3]]
        self.room[1] = [self.wall_v[1]] + [self.wall_v[3]] + [self.wall_h[2]] + [self.wall_h[3]]
        self.room[2] = [self.wall_v[3]] + [self.wall_v[4]] + [self.wall_h[2]] + [self.wall_h[3]]
        self.room[3] = [self.wall_v[0]] + [self.wall_v[4]] + [self.wall_h[1]] + [self.wall_h[2]]
        self.room[4] = [self.wall_v[0]] + [self.wall_v[2]] + [self.wall_h[0]] + [self.wall_h[1]]
        self.room[5] = [self.wall_v[2]] + [self.wall_v[4]] + [self.wall_h[0]] + [self.wall_h[1]]

        self.door_gap = {1:[-3,-2,3],
                         2:[1,2,3],
                         3:[5,6,3],
                         5:[-1,0,-1],
                         6:[5,6,-1]}
        self.in_room = self.Which_Room()
    def Which_Room(self):
        if self.px>-6 and self.px<6:
            if self.py>-6 and self.py<6:
                if self.py>-6 and self.py<-1:
                    if self.px>-6 and self.px<0:
                        return 5
                    elif self.px>0 and self.px<6:
                        return 6
                elif self.py>-1 and self.py<3:
                    return 4
                elif self.py>3 and self.py<6:
                    if self.px>-6 and self.px<-2:
                        return 1
                    elif self.px>-2 and self.px<2:
                        return 2
                    elif self.px>2 and self.px<6:
                        return 3
                elif self.py==-1:
                    if self.px>-2 and self.px<0:
                        return 5.5
                    elif self.px>4 and self.px<6:
                        return 6.5
                elif self.py==3:
                    if self.px>-4 and self.px<-2:
                        return 1.5
                    elif self.px>0 and self.px<2:
                        return 2.5
                    elif self.px>4 and self.px<6:
                        return 3.5
        return 0
    def wall_in_place(self):
        if int(self.in_room) == self.in_room and self.in_room != 0:
            return self.room[self.in_room-1]
        elif int(self.in_room) != self.in_room:
            pass
    def point_distance(self,point1,point2):
        return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    def min_distance(self):
        if int(self.in_room) == self.in_room and self.in_room != 0:
            if self.in_room == 4:
                min_dis = []
                ignore = []
                for key,value in self.door_gap.items():
                    if self.px > value[0] and self.px <value[1]:
                        min_dis.append(self.point_distance([self.px,self.py],[self.door_gap[key][0],self.door_gap[key][2]]))
                        min_dis.append(self.point_distance([self.px,self.py],[self.door_gap[key][1],self.door_gap[key][2]]))
                        ignore.append(self.door_gap[key][2])
                if self.wall_h[1][0][1] not in ignore:
                    min_dis.append(np.abs(self.py-self.wall_h[1][0][1]))
                if self.wall_h[2][0][1] not in ignore:
                    min_dis.append(np.abs(self.py-self.wall_h[2][0][1]))
                return min(min_dis)

            elif self.px < self.door_gap[self.in_room][0]:
                return min(np.abs(self.px-self.wall_in_place()[0][0][0]),np.abs(self.px-self.wall_in_place()[1][0][0]),
                           np.abs(self.py-self.wall_in_place()[2][0][1]),np.abs(self.py-self.wall_in_place()[3][0][1]))
            else:
                return min(np.abs(self.px-self.wall_in_place()[1][0][0]),np.abs(self.px-self.wall_in_place()[3][0][0]),self.point_distance([self.px,self.py],[self.door_gap[self.in_room][0],self.door_gap[self.in_room][2]]))
        elif int(self.in_room) != self.in_room:
            return min(np.abs(self.px-self.door_gap[int(self.in_room)][0]),np.abs(self.px-self.door_gap[int(self.in_room)][1]))
        else:
            return -1



