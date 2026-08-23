import numpy as np
from.demo_MSAA import AntiAliasRenderer
from scipy.interpolate import splprep, splev
class ParametricDemo:
    """
    这个类通过输入控制点坐标即可生成对应的参数化曲线
    """
    def __init__(self, curve_type:str, mask_template:np.ndarray):
        # self.cps = cps
        self.curve_type = curve_type
        self.mask_template = mask_template
        self.render = AntiAliasRenderer(msaa_level=16)
        # self.parametric_mask = self.render_curve()
        
    def render_curve(self,cps):
        """ 
        根据控制点的位置进行曲线渲染
        """
        if self.curve_type == 'OA':
            mask = self.render.MSAA(cps, self.mask_template)
        if self.curve_type == 'BZ':
            BZ_cps = self.compute_BZCPandBZ_points(cps,self.mask_template)
            mask = self.render.MSAA(BZ_cps, self.mask_template)
        if self.curve_type == 'BS':
            BS_cps = self.b_spile(cps, 0.3, num_points=200)
            mask = self.render.MSAA(BS_cps, self.mask_template)
        return mask

    def get_curve_points(self, cps, num_points=200):
        """
        仅提取曲线点坐标 (不渲染为像素 mask)。
        返回 list of (N_i, 2) ndarray [y, x]。

        Parameters
        ----------
        cps : list of ndarray  每条轮廓的控制点
        num_points : int       B 样条采样点数 (仅 BS 模式生效)

        Returns
        -------
        curves : list of (N_i, 2) ndarray  每条轮廓的曲线离散点 [y, x]
        """
        if self.curve_type == 'OA':
            return [np.asarray(c, dtype=np.float64) for c in cps]
        if self.curve_type == 'BZ':
            return self.compute_BZCPandBZ_points(cps, self.mask_template)
        if self.curve_type == 'BS':
            return self.b_spile(cps, 0.3, num_points=num_points)
        return cps
        
    def b_spile(self, contours, smoothing=0.3, num_points=100, save_path=None):
        fitted_contours = []
        for contour in contours:
            contour = np.array(contour, dtype=np.float64)  # 强制转换为 float64
            y, x = contour[:, 0], contour[:, 1]

        # 拟合 B 样条
            tck, u = splprep([x, y], s=smoothing, per=True)  # 周期性轮廓可设 per=True
            u_fine = np.linspace(0, 1, num_points)
            x_fit, y_fit = splev(u_fine, tck)

        # 保存拟合后的轮廓点
            fitted = np.stack([y_fit, x_fit], axis=1)
            fitted_contours.append(fitted)

        return fitted_contours
    
    def generate_bezier_tangent(self, contours, mask):
        approxs_tangent = []
        height, width = mask.shape
        num = 1
        for contour in contours:
            # approx = contour.reshape(-1, 2).astype(np.float64)
            contour = np.array(contour)
            approx = contour.reshape(-1, 2).astype(np.float64)
           
            # 将 [y, x] 格式转换为 [x, y] 格式（便于向量计算）
            approx_xy = approx[:, [1, 0]]  # 交换列，得到 [x, y] 顺序
            
            P_prev = np.roll(approx_xy, 1, axis=0)  # 前一个点（[x_prev, y_prev]）
            P_next = np.roll(approx_xy, -1, axis=0)  # 后一个点（[x_next, y_next]）
            n = len(approx_xy)
            contour_point_tangent = []
            
            for i in range(n):
                A = P_prev[i]  # 前一个点 [x_prev, y_prev]
                B = approx_xy[i]  # 当前点 [x_curr, y_curr]
                C = P_next[i]  # 后一个点 [x_next, y_next]
            
                # 计算前向和后向向量（基于 [x, y] 坐标）
                v_prev = (A - B) / (np.linalg.norm(A - B) + 1e-8)  # 防止除零
                v_next = (C - B) / (np.linalg.norm(C - B) + 1e-8)
            
                # 计算角平分线（v_prev + v_next）
                bisector = v_prev + v_next
            
                # 逆时针旋转90度（原向量 [dx, dy] → 旋转后 [-dy, dx]）
                bisector_rotated = np.array([-bisector[1], bisector[0]])  # 修正旋转方向
            
                # 归一化得到切线方向
                norm = np.linalg.norm(bisector_rotated)
                if norm >= 1e-6:
                    tangent = bisector_rotated / norm
                else:
                    # 备用方案：使用后向/前向向量方向
                    if np.linalg.norm(v_next) > 1e-6:
                        tangent = v_next / np.linalg.norm(v_next)
                    else:
                        tangent = v_prev / np.linalg.norm(v_prev)
            
                # 确保切线方向与后向向量方向一致（点积判断）
                if np.dot(tangent, v_next) < 0:
                    tangent = -tangent
            
                # 将切线方向从 [x, y] 转回 [y, x] 格式（与输入一致）
                tangent_yx = np.array([tangent[1], tangent[0]])
                
                contour_point_tangent.append({
                    'point': approx[i],
                    'tangent': tangent_yx})
            
            approxs_tangent.append({
                'contour_point_tangent': contour_point_tangent,
                'data_point':contour,
                'height':height,
                'width':width
                
                    })
        
        return approxs_tangent
    def cubic_bezier(self,p0, p1, p2, p3, num):
        t = np.linspace(0, 1, num)[:, None]
        curve = (1 - t) ** 3 * p0 + 3 * t * (1 - t) ** 2 * p1 + 3 * t ** 2 * (1 - t) * p2 + t ** 3 * p3
        return curve
    
    def generate_BZ_points(self, all_contours):
        bezier_curve = []

        for contour in all_contours:
            controlpoints = contour['control_points']
            # 用列表收集每一段曲线的 numpy 数组
            curve_segments = []
            for cp in controlpoints:
                p0, p1, p2, p3 = cp
                curve = self.cubic_bezier(p0, p1, p2, p3, num=10)  # 返回 shape=(num,2) 的数组
                curve_segments.append(curve)

            # 一次性拼接成一个轮廓的完整曲线
            poly = np.vstack(curve_segments).astype(np.float32)
            bezier_curve.append(poly)

        return bezier_curve
    
    def compute_BZCPandBZ_points(self, initial_cps,initial_mask):
        points_tangent = self.generate_bezier_tangent(initial_cps,initial_mask)
        segments = []
        num = 1
        contourpoint = []
        for contour_point in points_tangent:
            # print(contour_point)
            height = contour_point['height']
            width = contour_point['width']
            data_point = contour_point['data_point']
            contour = contour_point['contour_point_tangent']
            n = len(contour)
            contour_control_points = []
            contour_control_point0 = []
            for i in range(n):
                p0 = np.array(contour[i]['point'])
                t0 = np.array(contour[i]['tangent'])
                if i < n -1 :
                    p3 = np.array(contour[i+1]['point'])
                    t3 = np.array(contour[i+1]['tangent'])
                else:
                    p3 = np.array(contour[0]['point'])
                    t3 = np.array(contour[0]['tangent'])
                d = np.linalg.norm(p3 - p0)
                alpha = beta = d * 0.2
                p1 = p0 + alpha * t0
                p2 = p3 - beta * t3
                contour_control_point0.append([p0,p1,p2,p3])
                # contour_control_points.append({'p0':p0,
                                            #    'p1':p1,
                                            #    'p2':p2,
                                            #    'p3':p3})
                
            # segments.append(contour_control_points)  #存储的是带有名称的形式
            
            contourpoint.append({'data_points':data_point,
                                 'curve_type':'bezier',
                                 'degree':3,
                                 'image_width': width,
                                 'image_height': height,
                                 'control_points':contour_control_point0#表示的是一个数组里面有4个点的坐标
                                 
                                 })
            # filename = './bezier/bezier_control_point/contour_{}.txt'.format(num)      
            # self.write_nurbs_file(filename, contourpoint)
            num += 1
            
        
        
        
        # print(contourpoint)
        BZ_points = self.generate_BZ_points(contourpoint)  
        num = 1
            
        return BZ_points   