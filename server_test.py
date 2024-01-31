#from flask import Flask, request, jsonify, render_template
import os
import ezdxf
import matplotlib.pyplot as plt
import math
import datetime 
import numpy as np 
from collections import defaultdict
import svgwrite
from svgwrite import px
import shutil

import csv
import pandas as pd
from shapely.geometry import Polygon, Point


#определяем функции

def read_dxf_entities(file_path):       #читаем файл dxf и разбираем его на сущности
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    #extmin, extmax = msp.query('EXTMIN', 'EXTMAX')
    entities = [entity for entity in msp if entity.dxftype()]
    
    return entities #, extmax, extmin


#функции обработки dxf

def distance_between_points(p1, p2):
     # Ensure that p1 and p2 are iterables and have at least 2 components
    if isinstance(p1, ezdxf.acc.vector.Vec3):
        p1 = p1.xyz  # конвертировать Vec3 в кортеж (x, y, z)
    if isinstance(p2, ezdxf.acc.vector.Vec3):
        p2 = p2.xyz
    
    p1, p2 = np.array(p1[:2]), np.array(p2[:2])
    
    return np.linalg.norm(p2 - p1)
    

def point_on_arc(cx, cy, radius, angle):
    
    angle_rad = math.radians(angle)
    x = cx + radius * math.cos(angle_rad)
    y = cy + radius * math.sin(angle_rad)
    
    return x, y

def arc_to_lines(arc):
    """Преобразует ARC в набор отрезков"""
    start_angle, end_angle = arc.dxf.start_angle, arc.dxf.end_angle
    cx, cy, _ = arc.dxf.center
    radius = arc.dxf.radius
    start_point = point_on_arc(cx, cy, radius, start_angle)
    end_point = point_on_arc(cx, cy, radius, end_angle)
    #print('arc_to_lines=', start_point, end_point)

    return [(start_point, end_point)]


def get_circle_segments(x, y, r, n_segment): 
    """функция для преобразование круга во многогранник"""
    points = []
    segments = []
    for i in range(n_segment):
        angle = 2 * math.pi * i / n_segment  # Угол в радианах
        x_point = x + r * math.cos(angle)
        y_point = y + r * math.sin(angle)
        points.append((round(x_point,2), round(y_point,2)))
            
    for i in range(0, n_segment-1):
        segments.append((points[i], points[i+1])) 
    segments.append((points[n_segment-1], points[0]))
    
    return segments    
 

def boundary_point_collector(out_entite_points, dict_elements): #получаем список id внешних закрытых контуров. Формируем списко крайних точек наброска

        

        #здесь мы должны перебрать все точки внешних контуров и выбрать экстремумы
    extrems = [float('inf'), float('-inf'), float('inf'), float('-inf')] #min_x, max_x, min_y, max_y

    for point_set in out_entite_points:
        points = dict_elements[point_set]

        for point in points:        #по идее можно заменить на более простую и быструю функцию
                
            if point[0] < extrems[0]:
                extrems[0] = point[0]
                    #outerline_object_ID.append(entite_id)
            if point[0] > extrems[1]:
                extrems[1] = point[0]
                    #outerline_object_ID.append(entite_id)
            if point[1] < extrems[2]:
                extrems[2] = point[1]
                    #outerline_object_ID.append(entite_id)
            if point[1] > extrems[3]:
                extrems[3] = point[1]
                    #outerline_object_ID.append(entite_id)
            #print(extrems)
        
        #print('outerline_object_ID',outerline_object_ID)
        #print('extrems = ', extrems, '\n')
        
    return  extrems

    
def dxf_entities_to_dict(entities):
        
    open_paths = []
    total_length = 0
    entites_dict = {} #словарь сущность:точки
    segments_dict = {} #словарь сущность:точки
    segments = []
        

    for e in entities:
        segments = []       #начинаем собирать сегменты для данной сущности
        raw_segments = []   #не округлённые отрезки сегмента
        entit_id = e.dxf.handle #сразу берём  ид сущности
        points_list = [] #пустой список точек для словаря
        #проверка и разборка сущности на точки в звависимости каким она объектом является
        #print('e.dxftype()', e.dxftype()) вывод типов сущностей
        if e.dxftype() == 'LINE':
            a1 = tuple(e.dxf.start)
            b1 = tuple(e.dxf.end)
            
        
            a = a1[0:2]
            b = b1[0:2]
            #print('a,b=', a, b)

            segments.append((a, b))#(round(a,2), round(b,2))
                
            #print('segments LINE', segments)
                
        elif e.dxftype() == 'CIRCLE': #если кольцо, то преобразуем к замкнутому многограннику
            #circles.append(e)   
            cx, cy, _ = e.dxf.center
            r = e.dxf.radius
                 
            num_segment = 12
            segment_points = get_circle_segments(cx, cy, r, num_segment)
                
            segments.extend(segment_points)
            #print('segments CIRCLE', segments)
                
        elif e.dxftype() == 'ARC':
            segments_from_arc = arc_to_lines(e)
            segments.extend(segments_from_arc)
            #print('segments ARC', segments)

        elif e.dxftype() == 'LWPOLYLINE':
            with e.points() as pts:
                #print('pts==', pts, type(pts))
                raw_points = [p[:2] for p in pts]
                #print('raw_points LWPOLYLINE', raw_points)
                points = []
                for elem in raw_points:
                     points.append((round(elem[0],2),round(elem[1],2)))
            # Get only x, y coordinates
            for i in range(len(points)-1):
                segments.append((points[i], points[i+1]))
            if e.is_closed:
                segments.append((points[-1], points[0]))
                #print('segments LWPOLYLINE', segments)
            else:
                open_paths.extend([(points[i], points[i+1]) for i in range(len(points)-1)])
                #print('open patch =', open_paths)
                
        elif e.dxftype() == 'SPLINE':
            fit_points = e.fit_points
            
            if fit_points:  # if fit_points is not empty
                for i in range(len(fit_points) - 1):
                    segments.append((fit_points[i][0:2], fit_points[i+1][0:2]))
                    
            else:
        # handle splines defined by control points
                control_points = e.control_points
                for i in range(len(control_points) - 1):
                    first_elem = control_points[i][0:2]
                    r_first_elem = tuple(round(num, 2) for num in first_elem)
                    second_elem = control_points[i+1][0:2] 
                    r_second_elem = tuple(round(num, 2) for num in second_elem)
                    #seg = (r_first_elem, r_second_elem)

                    segments.append((r_first_elem, r_second_elem))
                    #print('segments', seg)
                    
                    
            print('segments SPLINE', segments)
        #print('entit_id==',entit_id)
        for i in range(len(segments)):  #это мы точки извлекаем
            points_list.append(segments[i][0])
            #print('\n i=',i,'point=', segments[i][0],'\n')
        #print('points list=', points_list,'\n')
        
        segments_dict.update({entit_id:segments})

        entites_dict[entit_id] = points_list    #добавляем элемент в словарь контуров по принципу {ИД:ТОЧКИ}

        for start, end in segments: #считаем длинну сегмента и скидываем её в общую длинну
            length = round(distance_between_points(start, end),2)
            total_length += length
        
        #print('segments =', segments)
            
    return entites_dict, total_length, segments_dict
        
def find_closed_paths(entites_dict):
    completed_paths = [] #список из отрезков замкнутого контура
    closed_countur = {} #словарь замкнутых контуров
    
    for key, value2 in entites_dict.items():
            
        segment_dict = defaultdict(list)
        for s in value2:
            segment_dict[s[0]].append(s)
            segment_dict[s[1]].append((s[1], s[0]))  # Обратный сегмент
            #print('segment_dict', segment_dict)

        value = value2.copy() #для того, чтоб исходный словарь не убился    
        while value:
            current_path = [value.pop(0)]
            #print('current patch', current_path)
            while True:
                last_point = current_path[-1][1]
                next_segment = None
                for s in segment_dict[last_point]:
                    if s not in current_path and (s[1], s[0]) not in current_path:
                        next_segment = s
                        break

                if next_segment:
                    if next_segment in value:  # Добавляем эту проверку
                           value.remove(next_segment)
                    current_path.append(next_segment)
                    if next_segment[0] != next_segment[1]:
                        segment_dict[next_segment[0]].remove(next_segment)
                        segment_dict[next_segment[1]].remove((next_segment[1], next_segment[0]))
                else:
                    break

        if current_path[0][0] == current_path[-1][1]: #проверяем, равно ли начало отрезка его концу, если да, то он закрытый 
            completed_paths.append(current_path)
            closed_countur.update({key:current_path})
            #print('Completed patch find!', 'id=',key, current_path,'\n\n')
                
        
    return  len(completed_paths), closed_countur#, open_paths #! сколько и кто не замкнут


def in_polygon_test(base_polygon_id, tested_id, all_polygon_dict): #point по идее - сюда вкинуть ид подопытного
    base_level = 0 #ставим флаг базового состояния, 0 - внешний иначе - внутренний к базовому
    # Определение проверяемого полигона по его ИД
    #print('PIG=',all_polygon_dict, base_polygon_id, tested_id) #, 
    polygon = Polygon(all_polygon_dict[base_polygon_id])#[(0, 0), (1, 1), (1, 0)])

    # Определение точки
    for point in all_polygon_dict[tested_id]:
        #print('test point', point)
        test_point = Point(point)#0.35, 0.15)

        # Проверка, находится ли точка внутри полигона
        inside = test_point.within(polygon)
        #print(inside)
        if inside == True:
            base_level += 1
            #print('inside true!',base_polygon_id, tested_id)
    #print('base_level', base_level)

    return base_level

def main_countur_detector(full_polygon_dict):
    main_countur_list = [] #список внешних контуров
    all_countur_list = [] #сюда закидываем все id контуров
    inner_countur = []
    for item in full_polygon_dict:
        #print('Item ID = ', item, '\n')
        all_countur_list.append(item)

    for i in range(len(all_countur_list)):


    # Сравниваем элемент i с каждым другим элементом
        for j in range(len(all_countur_list)):
            if i != j:
                #print('all_countur_list[i]', all_countur_list[i])
                state = in_polygon_test(all_countur_list[i], all_countur_list[j], full_polygon_dict)
                #отправляем по очереди все элементы в поределение вложенности. Если второй элемент вложен в первый, ту удаляем его из смписка

                if state > 0:
                    inner_countur.append(all_countur_list[j])

    main_countur_list = [i for i in all_countur_list if i not in inner_countur]
    return main_countur_list 


def dxf_to_svg(entities, output_file, box): #собираем svg
        print('box',box)
        # Рассчитайте центр viewbox  -  рассчитаем по экстремумам
        center_x = (box[0] + box[1]) / 2
        center_y = (box[2] + box[3]) / 2
            # Создайте SVG

        
        
        #aa = round(box[0],2) 
        #bb = round(box[1],2)  
        #cc = round(box[2],2)  
        #dd = round(box[3],2)

           
        #print('a,b,c,d=', aa, bb, cc, dd,'\n')

        width = (box[1] - box[0]) * 1.0,
        height = (box[3] - box[2]) * 1.05
        dwg = svgwrite.Drawing(output_file, profile='tiny', size=(380,190), stroke_width=(abs(box[1])+abs(box[3]))/500) #,((abs(bb)+abs(dd))/200) viewBox=f'{aa}, {bb}, {cc}, {dd}size=('100%', '100%')(cc-aa, dd-bb) тут херня какая-то =f'{extmin[0]} {extmin[1]} {extmax[0] - extmin[0]} {extmax[1] - extmin[1]}'
        
        dwg.viewbox(
        minx = 0,#extmin[0], #extmin[0] - center_x,
        miny = box[2], #extmin[1] - center_y,
        width = width,
        height = height

        
)     
        #area = width * height
        print(f"Bounding Box Dimensions: Width: {width}, Height: {height}")       
        g = svgwrite.container.Group()
        #g.translate(-center_x, -center_y)  # Смещение объектов к центру viewbox

        # Добавьте объекты в группу
        for entity in entities:
            if entity.dxftype() == 'LINE':
                g.add(dxf_line_to_svg(entity))
            elif entity.dxftype() == 'CIRCLE':
                g.add(dxf_circle_to_svg(entity))
            elif entity.dxftype() == 'ARC':
                g.add(dxf_arc_to_svg(entity))
            elif entity.dxftype() == 'LWPOLYLINE':
                g.add(dxf_lwpolyline_to_svg(entity))
            elif entity.dxftype() == 'SPLINE':
                g.add(dxf_spline_to_svg(entity))
                
                
        #скрипт трансформации / сдвига изображения, когда оно нарисовано фиг знает где
        x_shift = 0
        y_shift = 0

        if box[0] < 0:
            x_shift = -box[0] + 1
        elif box[0] >= 1:
            x_shift = -box[0]
        print('x_shift', x_shift) 
        
        if box[1] < 0:
            y_shift = -box[1] + 1
        elif box[1] >= 1:
            y_shift = -box[1]
        print('y_shift', y_shift)

        # Добавьте группу к SVG
        g['transform'] = f'scale(1, -1) translate({x_shift}, {y_shift - box[3]-1 })'#{x_shift}, {y_shift - dd-5 }
        dwg.add(g)

        # Сохраните SVG
        dwg.save()


def dxf_line_to_svg(line):
    x1, y1 = line.dxf.start.x, line.dxf.start.y
    x2, y2 = line.dxf.end.x, line.dxf.end.y
    return svgwrite.shapes.Line(start=(x1, y1), end=(x2, y2), stroke=svgwrite.rgb(0, 0, 0, '%'), )

        # Преобразование объекта CIRCLE в SVG
def dxf_circle_to_svg(circle):
    cx, cy = circle.dxf.center.x, circle.dxf.center.y
    r = circle.dxf.radius
    return svgwrite.shapes.Circle(center=(cx, cy), r=r, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none')

        # Преобразование объекта ARC в SVG
def dxf_arc_to_svg(arc):
    start_angle = arc.dxf.start_angle
    end_angle = arc.dxf.end_angle
    cx, cy = arc.dxf.center.x, arc.dxf.center.y
    r = arc.dxf.radius

    path = svgwrite.path.Path()
    path.push(['M', (cx + r * np.cos(np.radians(start_angle)), cy + r * np.sin(np.radians(start_angle)))])
    path.push(['A', r, r, 0, 0, 0, (cx + r * np.cos(np.radians(end_angle)), cy + r * np.sin(np.radians(end_angle)))])
    path.stroke = svgwrite.rgb(0, 0, 0, '%')
    path.fill = 'none'

    return path

        # Преобразование объекта LWPOLYLINE в SVG
def dxf_lwpolyline_to_svg(lwpolyline):
    points = [(x, y) for x, y in lwpolyline.get_points('xy')]
    return svgwrite.shapes.Polyline(points, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none')

        # Преобразование объекта SPLINE в SVG
def dxf_spline_to_svg(spline):
    if spline.fit_points:
                points = [(x, y) for x, y, _ in spline.control_points]
    elif spline.control_points:
                points = [(x, y) for x, y, _ in spline.control_points]
        

    return svgwrite.shapes.Polyline(points, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none')
    

    
def calculate(filepath):

    #сам скрипт
    print('start')
    #копируем файл во временный и смело его ломаем
    #uploaded_file = request.files['vectorFile']
    #orig_filepath = uploaded_file.filename  # путь к файлу
    #filepath = 'temp_'+ orig_filepath
    #shutil.copyfile(orig_filepath, filepath)

    #uploaded_file.save(filepath)
    

    dxf_entities = read_dxf_entities(filepath) #получили набор сущностей из dxf
    print(dxf_entities)
    dict_elements, total_length, segment_elem = dxf_entities_to_dict(dxf_entities) # и здесь мы должны полчить набор сегментов
    print('dict_elements', dict_elements)
    pin_holes, closed_countur = find_closed_paths(segment_elem) #, open_patch
    print('dict_elements', closed_countur)
    out_list = main_countur_detector(dict_elements) # dict_elements  закидываем сюда весь словарь

    extrems = boundary_point_collector(out_list, dict_elements) #вычисляем крайние точки внешних контуров

    

    date = datetime.datetime.now().strftime('%H-%M-%S_%Y-%m-%d')

    output_file = f'C:/test/{date}.svg'

    dxf_to_svg(dxf_entities, output_file, extrems)
    
    #print('Словарь контуров = dict_elements => ', dict_elements, '\nЕго длинна', len(dict_elements), 'элемента\nДлинна контуров = ', round(total_length,2))
    #print('segment_elem=', segment_elem)
    
    #print('Словарь контуров = dict_elements,  ', all_dict_el, '\nЕго длинна', len(dict_elements), 'элемента\nДлинна контуров = ', round(total_length,2))
    print('\nPin holes =', pin_holes, '\nВнешние контура', out_list, '\nДлинна контуров = ', round(total_length,2) , '\nКрайние точки = ', extrems) # closed_countur
    
    
    #area = S1+S2+Sn площадей внешних контуров


    #return size, area, total_length, pin_holes, svg_patch #то что мы возвращаем в веб 
    """return jsonify(
        puncture = circles + closed_path_count,
        total_length = round(total_length, 2),
        widths = round(width, 2),
        heights = round(height, 2),
        cost = ((total_length / 1000) * 2),""
                    
                    

        )#, render_template('calculator.html')"""

    #if __name__ == '__main__':
    #    app.run(debug=True , host="192.168.1.69")
    #@app.route('/materials', methods=['GET'])
    #def materials():
    #    return jsonify(MATERIALS)

    #@app.route('/get_pic/', methods=['POST'])
    #def get_pic():

    #    return jsonify(img = img)

if __name__ == '__main__':
    dxf_file_path = 'C:/test/центральная часть.dxf'  # Путь к вашему DXF файлу
    calculate(dxf_file_path)
    #1Квадрат и дырки2
    #Christmas E0018206 file cdr and dxf free vector download for laser cut
    #центральная часть
    #Zakaz1/911.0.01.06.01.02_Кронштейн_подвески