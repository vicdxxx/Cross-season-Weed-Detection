from numpy import newaxis


def list_dir(path, list_name, extension):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                list_name.append(file_path)
    return list_name


def simple_plot(data):
    from matplotlib import pyplot as plt
    plt.plot(data)
    plt.show()


def show_data_as_image(x):
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    if isinstance(x, list):
        x_list = x
    else:
        sample_num = len(x)
        if sample_num > 100:
            x_list = [x]
        else:
            x_list = x
    sample_num = len(x_list)
    idx = 0
    if isinstance(x_list[0], float) or isinstance(x_list[0], int):
        plt.plot(x_list)
    else:
        for x in x_list:
            if isinstance(x, np.ndarray):
                y = x.copy()
            else:
                import torch
                if isinstance(x, torch.Tensor):
                    y = x.cpu().detach().numpy().copy()
            y = y.squeeze()
            #y = (y - y.min())/(y.max()-y.min())
            if sample_num > 1:
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                plt.subplot(sample_num, 1, idx+1)
                idx += 1
            if len(y.shape) == 2:
                plt.imshow(y)
                plt.colorbar()
            elif len(y.shape) == 1:
                plt.plot(y)
    plt.show()


def fake_logger():
    if log_xalg is not None:
        class log_tmp():
            pass

        def empty_func(*info):
            pass
        logger = log_tmp()

        setattr(logger, "info", log_xalg)
        setattr(logger, "error", log_xalg)
        setattr(logger, "debug", empty_func)
        setattr(logger, "set_logger_with_File_handler", log_xalg)
        setattr(logger, "clear_logger_with_File_handler", log_xalg)
        return logger


def make_dir(file_path):
    import os
    if not os.path.isfile(file_path):
        f_log = open(file_path, 'w')
        f_log.close()
    else:
        f_log = open(file_path, 'a')
        f_log.close()
    return file_path


def log_xalg(*info, log_path=None, show=True, end=None):
    if show:
        if end is not None:
            print(*info, end=end)
        else:
            print(*info)
    if log_path:
        f_log = open(log_path, 'a')
        print(*info, file=f_log)
        f_log.close()


def load_txt_one_digit_per_line(path):
    res = []
    with open(path, 'r') as f:
        txt = f.readlines()
        for line in txt:
            res.append(float(line.split('\n')[0]))
    return res


"""
import torch
field = torch.tensor(
    [
        [
            [0.0654, 0.0250],
            [0.1746, 0.0431],
            [0.0978, -0.0130],
            [0.0751, 0.0065],
            [0.1252, -0.0041],
            [0.1094, 0.0260],
            [0.0441, 0.0066],
            [0.0224, 0.0012],
            [0.0526, 0.0055],
            [0.1179, -0.0076]
        ]
    ], device='cuda:0')

xys = torch.tensor(
    [
        [
            [0.3151, -0.2907],
            [-0.2990, -0.2880],
            [0.0154, -0.8878],
            [-0.2809, 0.0782],
            [0.0947, 0.8730],
            [0.1787, 0.3668],
            [-0.1151, 0.7378],
            [0.1626, -0.2425],
            [0.2818, 0.0809],
            [0.0226, -0.6917]
        ]
    ], device='cuda:0')
# show_flow_field(field)
show_flow_field(field, xys, shape=[58, 58])
"""
def show_flow_field(flow_field, xys=None, shape=None, scale=None, scale_units=None, headwidth=3, headlength=5, base_image=None, show=True):

    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib
    import numpy.ma as ma
    matplotlib.use('TkAgg')

    if isinstance(flow_field, np.ndarray):
        flow_field_np = flow_field.copy()
    else:
        import paddle
        if isinstance(flow_field, paddle.Tensor):
            flow_field_np = flow_field.numpy().copy()
        #import torch
        # if isinstance(flow_field, torch.Tensor):
        #    flow_field_np = flow_field.cpu().detach().numpy().copy()

    if xys is not None:
        if isinstance(flow_field, np.ndarray):
            xys_np = xys.copy()
        else:
            import paddle
            if isinstance(xys, paddle.Tensor):
                xys_np = xys.numpy().copy()
            # if isinstance(xys, torch.Tensor):
            #    xys_np = xys.cpu().detach().numpy().copy()
        xys_np = np.squeeze(xys_np)
        assert len(np.squeeze(flow_field_np).shape) == 2

    if len(flow_field_np.shape) == 3:
        flow_field_np = flow_field_np[np.newaxis, :]

    if len(flow_field_np.shape) == 4:
        if flow_field_np.shape[1] == 2:
            flow_field_np = np.transpose(flow_field_np, (0, 2, 3, 1))
        flow_field_np = flow_field_np[np.newaxis, :]

    if len(flow_field_np.shape) == 5:
        sample_num = min(flow_field_np.shape[0], 5)
        feature_num = min(flow_field_np.shape[1], 64)
    grid_size = int(feature_num**0.5+1)
    if shape is not None:
        assert xys is not None
        h, w = shape[:2]
    else:
        if len(flow_field_np.shape) == 5:
            h, w = flow_field_np.shape[2:4]
    print(flow_field_np.shape, sample_num, feature_num, grid_size, h, w)

    if base_image is None:
        Y = np.linspace(-1, 1, h)
        X = np.linspace(-1, 1, w)
    else:
        base_h, base_w = base_image.shape[:2]
        Y = np.linspace(1, base_h, h)
        X = np.linspace(1, base_w, w)
    X, Y = np.meshgrid(X, Y)
    for i_sample in range(sample_num):
        plt.figure(i_sample, figsize=(10, 10))
        plt.clf()
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i*grid_size+j
                if idx >= feature_num:
                    break
                # left=0.01, top=0.99, bottom=0.01, right=0.99
                # left=0.1, top=0.9, bottom=0.1, right=0.9
                if feature_num > 1:
                    plt.subplots_adjust(wspace=0.01, hspace=0.01)
                    plt.subplot(grid_size, grid_size, idx+1)
                plt.axis('off')
                if shape is not None:
                    U = np.ones((h, w)) * 1e-2
                    V = np.zeros((h, w))
                    unit_x = 2/w
                    unit_y = 2/h
                    for i_xy, xy in enumerate(xys_np):
                        x, y = xy
                        x_idx = round((x-(-1))/unit_x)
                        y_idx = round((y-(-1))/unit_y)
                        U[y_idx, x_idx] = flow_field_np[i_sample, :, i_xy, 0]
                        V[y_idx, x_idx] = -flow_field_np[i_sample, :, i_xy, 1]
                        #U[y_idx, x_idx] = 0.001
                        #V[y_idx, x_idx] = -0.001
                else:
                    U = flow_field_np[i_sample, idx, :, :, 0]
                    V = flow_field_np[i_sample, idx, :, :, 1]
                #mask = ((U > 0) + (V > 0)) > 0
                #U = ma.masked_array(U, mask=mask)
                #V = ma.masked_array(V, mask=mask)

                #M = (U**2+V**2)**0.5
                # width=0.005, scale=1
                print(X.shape)
                print(Y.shape)
                print(U.shape)
                print(V.shape)
                if base_image is not None:
                    plt.imshow(base_image)
                plt.quiver(X, Y, U, V, scale=scale, scale_units=scale_units, headwidth=headwidth, headlength=headlength, pivot='mid', color='r')
                # plt.gca().invert_yaxis()
    if show:
        plt.show()


def show_features(features, feature_dim_is_last=0, show_in_one_image=0, has_alpha_channel=0, show_color_image=0):
    # n_sample x n_features x f_h x f_w
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    if isinstance(features, np.ndarray):
        features_np = features.copy()
    else:
        #import paddle
        # if isinstance(features, paddle.Tensor):
        #    features_np = features.numpy().copy()
        import torch
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().detach().numpy().copy()
    if features_np.dtype == bool:
        features_np = features_np.astype(np.int) * 255
    if len(features_np.shape) == 2:
        features_np = features_np[np.newaxis, :]
    if len(features_np.shape) == 3:
        features_np = features_np[np.newaxis, :]
    if len(features_np.shape) == 4:
        features_np = features_np[np.newaxis, :]
    if len(features_np.shape) == 5:
        if feature_dim_is_last:
            features_np = features_np.transpose(0, 4, 1, 2, 3)
        if has_alpha_channel:
            features_np = features_np[..., :3]
        elif show_color_image:
            pass
        else:
            if features_np.shape[2] == 3 or features_np.shape[2] == 1:
                features_np = features_np.transpose(0, 1, 3, 4, 2)
            else:
                features_np = features_np[0]
                features_np = features_np[:, :, :, :, np.newaxis]
    print(features_np.shape)
    sample_num = min(features_np.shape[0], 5)
    feature_num = min(features_np.shape[1], 64)
    grid_size = int(feature_num**0.5+1)
    print(features_np.shape, sample_num, feature_num, grid_size)
    if show_in_one_image:
        im_show_all = np.zeros_like(features_np[0, 0])
    for i_sample in range(sample_num):
        plt.figure(i_sample, figsize=(10, 10))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i*grid_size+j
                if idx >= feature_num:
                    break
                # left=0.01, top=0.99, bottom=0.01, right=0.99
                # left=0.1, top=0.9, bottom=0.1, right=0.9
                im_show = features_np[i_sample, idx]
                if len(im_show.shape) == 3 and im_show.shape[-1] == 1:
                    im_show = im_show[:, :, 0]
                if show_in_one_image:
                    im_show_all += im_show
                else:
                    if feature_num > 1:
                        plt.subplots_adjust(wspace=0.01, hspace=0.01)
                        plt.subplot(grid_size, grid_size, idx+1)
                    im_show = (im_show - im_show.min()) / (im_show.max()-im_show.min())
                    plt.imshow(im_show)
                    plt.axis('off')
    if show_in_one_image:
        im_show = im_show_all
        im_show = (im_show - im_show.min()) / (im_show.max()-im_show.min())
        plt.imshow(im_show)
        plt.axis('off')
    # plt.tight_layout()
    plt.show()
    #plt.savefig('/mnt/e/PHD/BlueberryDenseDetection/Iter-E2EDET/output/debug.png')
    print('finished')


def show_image_simple(image, wait=True, name="x"):
    import numpy as np
    import cv2 as cv
    #img = image.cpu().detach().numpy().copy()
    if isinstance(image, np.ndarray):
        features_np = image.copy()
    else:
        #import paddle
        # if isinstance(features, paddle.Tensor):
        #    features_np = features.numpy().copy()
        import torch
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy().copy()
    img = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    img = img.astype(np.uint8)
    #cv.namedWindow(name, 0)
    cv.imshow(name, img)
    cv.waitKey()


def show_points(points, points_1):
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    x_list = []
    y_list = []
    for pt in points:
        x = pt['x']
        y = pt['y']
        x_list.append(x)
        y_list.append(y)
    plt.scatter(x_list, y_list, c='r')
    if points_1:
        x_list = []
        y_list = []
        for pt in points_1:
            x = pt['x']
            y = pt['y']
            x_list.append(x)
            y_list.append(y)
        plt.scatter(x_list, y_list, c='g')
    plt.show()


def show_image(image, wait=None, name="x", channel_reverse=True, points=None):
    import numpy as np
    import cv2 as cv
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        #import paddle
        #if isinstance(image, paddle.Tensor):
        #    img = image.numpy().copy()
        import torch
        if isinstance(image, torch.Tensor):
            img = image.cpu().detach().numpy().copy()

    if len(img.shape) == 4:
        if img.shape[1] == 3:
            img = img.transpose(0, 2, 3, 1)[0]
        if img.shape[1] == 1:
            img = image.transpose(0, 2, 3, 1)[0]
    elif len(img.shape) == 3:
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        elif img.shape[1] > img.shape[0] and img.shape[2] > img.shape[0]:
            img = img[0]
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    img = img.astype(np.uint8)
    img = np.squeeze(img)

    if points is not None:
        print("points num:", len(points))
        for point in points:
            point = (int(point[0]), int(point[1]))
            cv.circle(img, point, 3, (0, 255, 0), -1)
    if len(img.shape) == 3 and channel_reverse:
        img = img[:, :, ::-1]
    print(img.shape)
    cv.namedWindow(name, 0)
    cv.resizeWindow(name, 512, 512)
    cv.imshow(name, img)
    if wait is not None:
        cv.waitKey(wait)


def show_image_with_bbox(x, targets=None, colors=None, label_idx=0, use_cxcywh=1):
    tot_bbox_num = 0
    tot_im_num = 0

    tot_im_num += 1
    import cv2
    import numpy as np
    if isinstance(x, np.ndarray):
        y = x.copy()
    else:
        import torch
        if isinstance(x, torch.Tensor):
            if x.shape[1] == 3:
                x_new = x.permute(0, 2, 3, 1)[0]
                y = x_new.cpu().detach().numpy().copy()
            else:
                y = x.cpu().detach().numpy().copy()
    if y.max() <= 1.0:
        z = np.zeros(y.shape, dtype=np.float32)
        z = cv2.normalize(y, z, alpha=0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    else:
        z = y
    z = z.astype(np.uint8)
    print("image range: {} -> {}".format(z.min(), z.max()))
    z = np.ascontiguousarray(z, dtype=np.uint8)
    if targets is not None:
        if isinstance(x, np.ndarray):
            targets_y = targets
        else:
            if isinstance(targets, torch.Tensor):
                targets_ = targets.cpu().detach().numpy()
                targets_y = targets_[:, label_idx]
                xxx = targets_[:, 1]
                #targets_ = targets_[targets_y > 0]
                targets_ = targets_[xxx > 0]

        bbox_num = len(targets_)
        print("bbox_num: {}".format(bbox_num))
        tot_bbox_num += bbox_num
        for i_y in range(bbox_num):
            if label_idx == 0:
                sku_id = int(targets_[i_y, label_idx])
                bbox_start_idx = 1
                bbox_end_idx = 5
            elif label_idx == 4:
                sku_id = int(targets_[i_y, label_idx])
                bbox_start_idx = 0
                bbox_end_idx = 4

            if use_cxcywh:
                cx, cy, w, h = targets_[i_y, bbox_start_idx:bbox_end_idx]
                x0 = round(cx-w//2)
                y0 = round(cy-h//2)
                x1 = round(cx+w//2)
                y1 = round(cy+h//2)
            else:
                x0, y0, x1, y1 = targets_[i_y, bbox_start_idx:bbox_end_idx]

            if colors is not None:
                color = tuple(colors[sku_id])
            else:
                color = (255, 255, 255)
            thickness = 2
            font_scale = 1.0
            cv2.rectangle(z, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
            cv2.putText(z, str(sku_id), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    print("tot_bbox_num: {}".format(tot_bbox_num))
    if tot_im_num > 1:
        mean_bbox_num = tot_bbox_num / tot_im_num
        print("mean_bbox_num: {}".format(mean_bbox_num))

    cv2.namedWindow("show", 0)
    cv2.imshow("show", z)
    cv2.waitKey()


def show_lines(Image, Lines, name='x', wait=None):
    import cv2
    import numpy as np
    im = Image.copy()
    for i_line in range(len(Lines)):
        tmp = np.array(Lines).squeeze().astype(np.int)
        one_line = tmp[i_line]
        cv2.line(im, (one_line[0], one_line[1]), (one_line[2], one_line[3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.namedWindow(name, 0)
    cv2.imshow(name, im)
    if wait:
        cv2.waitKey(wait)


interactive_mode = 1
ax = None
verts_loop = None
first_time = True


def show_pointcloud_loop(arg=0):
    # slow 1 image/s
    global interactive_mode
    global ax
    global verts_loop
    global first_time
    from matplotlib import pyplot as plt
    surf = ax.scatter(verts_loop[:, 0] * 1.2, verts_loop[:, 1], verts_loop[:, 2], c='cyan', alpha=1.0, edgecolor='b')
    if not interactive_mode:
        if first_time:
            plt.show()
            first_time = False
        else:
            plt.show(block=False)
    if interactive_mode:
        # long time can operate graph, short time for quick show
        plt.pause(0.001)


#rot_show = utils.angle2matrix(np.array([90, 0, 0]))
#pt_3d_show = rot_show.dot(pt_3d.T).T
# if utils.first_time:
#    utils.show_pointcloud(pt_3d_show, use_pytorch3d=0, use_plt_loop=1, block=None, use_interactive_mode=1)
# else:
#    utils.verts_loop = pt_3d_show
#canvas = np.zeros((512, 512, 3))
#utils.show_image(canvas, points=pts2d_project, wait=1)
def show_pointcloud(verts, use_pytorch3d=1, use_plt_loop=0, block=None, use_interactive_mode=1):
    if use_pytorch3d:
        import torch
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        if not isinstance(verts, torch.Tensor):
            verts = torch.Tensor(verts).to(device)

        point_cloud = Pointclouds(points=[verts])
        point_cloud_person = {
            "3DMM": {
                "obj": point_cloud
            }
        }
        fig = plot_scene(
            point_cloud_person,
            xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            axis_args=AxisArgs(showgrid=True))
        fig.show()
    elif use_plt_loop:
        import pylab
        from matplotlib import pyplot as plt
        global interactive_mode
        global ax
        global verts_loop
        global first_time
        interactive_mode = use_interactive_mode
        if interactive_mode:
            plt.ion()
        import matplotlib
        matplotlib.use('TkAgg')
        fig = plt.figure(0, figsize=plt.figaspect(1.0))
        fig.canvas.mpl_connect('key_press_event', show_pointcloud_loop)
        verts_loop = verts
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev=0, azim=-90)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #ax.set_xlim(-1.0, 1.0)
        #ax.set_ylim(-1.0, 1.0)
        #ax.set_zlim(-1.0, 1.0)
        show_pointcloud_loop()
    else:
        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        surf = ax.scatter(verts[:, 0] * 1.2, verts[:, 1], verts[:, 2], c='cyan', alpha=1.0, edgecolor='b')
        plt.show()


def angle2matrix(angles, gradient='false'):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
        gradient(str): whether to compute gradient matrix: dR/d_x,y,z
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    import numpy as np
    from math import cos, sin
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    # R=Rx.dot(Ry.dot(Rz))

    if gradient != 'true':
        return R.astype(np.float32)
    elif gradient == 'true':
        # gradident matrix
        dRxdx = np.array([[0, 0, 0],
                          [0, -sin(x), -cos(x)],
                          [0, cos(x), -sin(x)]])
        dRdx = Rz.dot(Ry.dot(dRxdx)) * np.pi/180
        dRydy = np.array([[-sin(y), 0, cos(y)],
                          [0, 0, 0],
                          [-cos(y), 0, -sin(y)]])
        dRdy = Rz.dot(dRydy.dot(Rx)) * np.pi/180
        dRzdz = np.array([[-sin(z), -cos(z), 0],
                          [cos(z), -sin(z), 0],
                          [0, 0, 0]])
        dRdz = dRzdz.dot(Ry.dot(Rx)) * np.pi/180
        return R.astype(np.float32), [dRdx.astype(np.float32), dRdy.astype(np.float32), dRdz.astype(np.float32)]


def project_landmarks(camera_intrinsic, viewpoint_R, viewpoint_T, scale, headposes, pts_3d):
    ''' project 2d landmarks given predicted 3d landmarks & headposes and user-defined
    camera & viewpoint parameters
    '''
    rot, trans = angle2matrix(headposes[:3]), headposes[3:][:, None]
    pts3d_headpose = scale * rot.dot(pts_3d.T) + trans
    pts3d_viewpoint = viewpoint_R.dot(pts3d_headpose) + viewpoint_T[:, None]
    pts2d_project = camera_intrinsic.dot(pts3d_viewpoint)
    pts2d_project[:2, :] /= pts2d_project[2, :]  # divide z
    pts2d_project = pts2d_project[:2, :].T
    return pts2d_project, rot, trans


def show_histogram(data):
    import jittor as jt
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    if isinstance(data, jt.jittor_core.Var):
        data = data.numpy()
    my_dpi = 96
    plt.figure(figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)
    plt.hist(data, bins=100)
    plt.show()


def show_anchor_with_gt(pos_anchors=None, neg_anchors=None, gts=None, labels=None, im_name=None, im_dir=None, img_size=(1024, 1024, 3)):
    import jittor as jt
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon, Circle
    import matplotlib.patches as patches
    import numpy as np
    import os
    import cv2
    import math

    def cal_line_length(point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

    def get_best_begin_point_single(coordinate):
        x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)
        combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                    [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
        dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        force = 100000000.0
        force_flag = 0
        for i in range(4):
            temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                        + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                        + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                        + cal_line_length(combinate[i][3], dst_coordinate[3])
            if temp_force < force:
                force = temp_force
                force_flag = i
        if force_flag != 0:
            pass
            # print("choose one direction!")
        return np.array(combinate[force_flag]).reshape(8)

    def get_best_begin_point(coordinates):
        coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
        coordinates = np.array(coordinates)
        return coordinates

    def rotated_box_to_poly_np(rrects):
        """
        rrect:[x_ctr,y_ctr,w,h,angle]
        to
        poly:[x0,y0,x1,y1,x2,y2,x3,y3]
        """
        if rrects.shape[0] == 0:
            return np.zeros([0, 8], dtype=np.float32)
        polys = []
        for rrect in rrects:
            x_ctr, y_ctr, width, height, angle = rrect[:5]
            tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
            rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            poly = R.dot(rect)
            x0, x1, x2, x3 = poly[0, :4] + x_ctr
            y0, y1, y2, y3 = poly[1, :4] + y_ctr
            poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
            polys.append(poly)
        polys = np.array(polys)
        polys = get_best_begin_point(polys)
        return polys.astype(np.float32)

    wordname_FAIR1M_1_5 = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 'Football_Field', 'Baseball_Field',
            'Intersection', 'Roundabout', 'Bridge']
    classes = wordname_FAIR1M_1_5
    colors = (np.random.random((10, 3)) * 0.6 + 0.4).tolist()

    if im_dir is not None and im_name is not None:
        im_path = os.path.join(im_dir, im_name)
        img = cv2.imread(im_path)
        #cv2.imshow('x', img)
        #cv2.waitKey()
        img = img[:, :, ::-1]
    else:
        img = np.zeros(img_size)

    #if anchors is None:
    #    anchors = np.array([[180.576, 39.53623, 68.789734, 62.026207, 0.17472303]])
    #if gts is None:
    #    gts = np.array([[183., 40., 73.24827, 69.76814, -0.30720907]])

    my_dpi = 96
    fig = plt.figure(0, figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)
    plt.clf()
    plt.axis('off')
    ax = plt.gca()
    ax.imshow(img)

    ax.set_autoscale_on(False)

    if pos_anchors is not None:

        polygons = []
        color = []
        circles = []
        r = 5
        if isinstance(pos_anchors, jt.jittor_core.Var):
            pos_anchors = pos_anchors.numpy()
        polys = rotated_box_to_poly_np(pos_anchors)
        #c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        c = [50, 200, 50]
        for i_obj, poly in enumerate(polys):
            poly = poly.reshape(-1, 2)
            polygons.append(Polygon(poly))
            color.append(c)

            #label_idx = labels[i_obj]
            #c = colors[label_idx]
            color.append(c)

            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)

        #p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.3)
        #ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors='green', linewidths=1)
        ax.add_collection(p)
        #p = PatchCollection(circles, facecolors='red')
        #ax.add_collection(p)

    if neg_anchors is not None:
        polygons = []
        color = []
        circles = []
        r = 5
        if isinstance(neg_anchors, jt.jittor_core.Var):
            neg_anchors = neg_anchors.numpy()
        polys = rotated_box_to_poly_np(neg_anchors)
        c = [200, 50, 50]
        for i_obj, poly in enumerate(polys):
            #c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = poly.reshape(-1, 2)
            polygons.append(Polygon(poly))
            color.append(c)

            #label_idx = labels[i_obj]
            #c = colors[label_idx]
            color.append(c)

            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)

        #p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.3)
        #ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors='blue', linewidths=1)
        ax.add_collection(p)
        #p = PatchCollection(circles, facecolors='red')
        #ax.add_collection(p)

    if gts is not None:
        polygons = []
        color = []
        circles = []
        r = 5
        if isinstance(gts, jt.jittor_core.Var):
            gts = gts.numpy()
        polys = rotated_box_to_poly_np(gts)
        for i_obj, poly in enumerate(polys):
            #c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = poly.reshape(-1, 2)
            polygons.append(Polygon(poly))
            #color.append(c)
            color.append('red')
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)

        #p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.7)
        #ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=1)
        ax.add_collection(p)
        #p = PatchCollection(circles, facecolors='red')
        #ax.add_collection(p)
    plt.show()

    #t2 = mpl.transforms.Affine2D().rotate_deg(-45) + ax.transData
    #r2.set_transform(t2)

    #angle, float, Rotation in degrees anti-clockwise about xy.
    #if anchors is not None:
    #    for anchor in anchors:
    #        cx, cy, w, h, angle_radian = anchor
    #        cx = cx*np.cos(-angle_radian)-cy*np.sin(-angle_radian)
    #        cy = cx*np.sin(-angle_radian)+cy*np.cos(-angle_radian)

    #        patch = patches.Rectangle((cx-w/2, cy+h/2), w, h, angle=angle_radian/np.pi*180, color="blue", alpha=0.50)
    #        ax.add_patch(patch)
    #if gts is not None:
    #    for gt in gts:
    #        cx, cy, w, h, angle_radian = gt
    #        cx = cx*np.cos(-angle_radian)-cy*np.sin(-angle_radian)
    #        cy = cx*np.sin(-angle_radian)+cy*np.cos(-angle_radian)

    #        patch = patches.Rectangle((cx-w/2, cy+h/2), w, h, angle=angle_radian/np.pi*180, color="red", alpha=0.50)
    #        ax.add_patch(patch)
    #plt.show()


def showAnns(images, targets=None, outputs=None):
    import jittor as jt
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon, Circle
    import numpy as np
    """
    :param catNms: category names
    :param objects: objects to show
    :param imgId: img to show
    :param range: display range in the img
    :return:
    """

    wordname_FAIR1M_1_5 = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 'Football_Field', 'Baseball_Field',
            'Intersection', 'Roundabout', 'Bridge']
    classes = wordname_FAIR1M_1_5
    colors = (np.random.random((10, 3)) * 0.6 + 0.4).tolist()

    if isinstance(images, jt.jittor_core.Var):
        images = images.numpy()

    for i_img, img in enumerate(images):
        denormilize = True
        if denormilize:
            mean = [123.675, 116.28, 103.53]
            std = [58.395, 57.12, 57.375]
            mean = np.float32(mean).reshape(-1, 1, 1)
            std = np.float32(std).reshape(-1, 1, 1)
            img = img * std + mean
            img /= 255.0
        img = img[::-1]
        img = img.transpose(1, 2, 0)

        my_dpi = 96
        plt.figure(i_img, figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)
        plt.clf()
        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)


        if targets is not None:
            polygons = []
            color = []
            circles = []
            r = 3
            objects = targets[i_img]
            if 'classes' in objects:
                classes = objects['classes']
            #['rboxes', 'hboxes', 'polys', 'labels']
            if 'polys' in objects:
                labels = objects['labels']
                if isinstance(labels, jt.jittor_core.Var):
                    labels = labels.numpy()

                for i_obj, poly in enumerate(objects['polys']):
                    if isinstance(poly, jt.jittor_core.Var):
                        poly = poly.numpy().reshape(-1, 2)
                    polygons.append(Polygon(poly))

                    #c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

                    label_idx = labels[i_obj]
                    #label_idx = classes.index(label)
                    c = colors[label_idx]
                    color.append(c)

                    point = poly[0]
                    circle = Circle((point[0], point[1]), r)
                    circles.append(circle)

            #p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.7)
            #ax.add_collection(p)
            #p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=1)
            #ax.add_collection(p)
            #p = PatchCollection(circles, facecolors='red')
            p = PatchCollection(circles, facecolors=color)
            ax.add_collection(p)

        if outputs is not None:
            polygons = []
            color = []
            circles = []
            r = 2
            output = outputs[i_img]
            polys = output[0]
            labels = output[2]
            if isinstance(labels, jt.jittor_core.Var):
                labels = labels.numpy()

            for i_obj, poly in enumerate(objects['polys']):
                if isinstance(poly, jt.jittor_core.Var):
                    poly = poly.numpy().reshape(-1, 2)
                polygons.append(Polygon(poly))

                label_idx = labels[i_obj]
                #label_idx = classes.index(label)
                c = colors[label_idx]
                color.append(c)
                point = poly[0]
                #circle = Circle((point[0], point[1]), r)
                #circles.append(circle)
            #p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.7)
            #ax.add_collection(p)
            p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
            #p = PatchCollection(circles, facecolors='red')
            #ax.add_collection(p)
    plt.show()


def main():
    show_anchor_with_gt()
    pass


if __name__ == "__main__":
    main()
