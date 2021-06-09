import numpy as np
from wzk.mpl import (new_fig, save_fig, turn_ticks_off, get_pip, set_style,
                     geometry, figure, patches, golden_ratio,
                     get_aff_trafo, RelativeFancyArrow, FancyArrowX2, FancyBbox)

from mopla.Visualization import humanoid_robot_2d
from definitions import ICHR20_CALIBRATION_FIGS

# INFO for 2columns look at commit "Calibration Sketch 2 columns"  / 4ff8c36a
# INFO for 1columns look at commit "Calibration Sketch 1 column"  / ???

# Figure Dimensions
x0, y0 = .0, .0
x1 = 1.4
y1 = x1 * (np.sqrt(5.0) - 1.0) / 2.0

# Robot Configuration
head, chest, arms, hands, joints = humanoid_robot_2d.get_body_kwargs()
hands['coord_frame'] = 'diamond'

q_c = 0
# q_ra = np.array([+1.3, +2, 0.4])
# q_la = np.array([-1, -1.55, -0.45])
q_ra = np.array([+1.2, +1.55, 0.4])
q_la = np.array([-1.1, -2.1, -0.3])
q = np.hstack((q_c, q_ra, q_la))
robot_xy = np.array([x1*2/3, +.11])  # 2/3 for 2 columns

# Camera Eyes
cam1_xy = np.array([0.01, y1-0.01])
# cam1_xy = np.array([0.3, y1-0.2])
cam2_xy = np.array([x1*2/3, y1-0.01])
cam3_xy = np.array([x1-0.01, y1-0.01])

cam1_theta = np.deg2rad(-25)
cam2_theta = np.deg2rad(-90)
cam3_theta = np.deg2rad(222)

cam_r = 0.1
cam_arc = 0.5
cam_c = 'k'
cam_lw = 0.8

# Rays
ray_head_length = 0.05
ray_head_width = 0.02
ray_width = 0.005
ray_overhang = -0.1
ray1r_x = cam1_xy + cam_r * 1.3 * np.array([np.cos(cam1_theta), np.sin(cam1_theta)])


ray_color_valid = '#0065bd'  # tum blue_3
ray_color_orientation = '#d04592'
ray_color_occlusion = '#e37222'  # tum orange


def plot_ray(_ax, cam_x, target_x, color, offset0=cam_r*1.3, offset1=0.0):

    ray = FancyArrowX2(xy0=cam_x, xy1=target_x, offset0=offset0, offset1=offset1,
                       color=color, alpha=1, zorder=10,
                       width=ray_width, head_length=ray_head_length, head_width=ray_head_width,
                       overhang=ray_overhang,
                       length_includes_head=True)
    _ax.add_patch(ray)


def plot_ray_legend(_ax, x, y, dx, dy=0, color='k'):
    ray = patches.FancyArrow(x, y, dx, dy,
                             color=color, alpha=1, zorder=10,
                             width=0.01, head_length=dx/2, head_width=0.05,
                             overhang=ray_overhang,
                             length_includes_head=True)
    _ax.add_patch(ray)


# Legend
font_size = 8
ha = 'left'
va = 'center'

# Main
set_style(s=('ieee', 'no_borders'))
fig, ax = new_fig(aspect=1, width='ieee1c')
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
turn_ticks_off(ax=ax)


geometry.eye_pov(ax=ax, xy=cam1_xy, angle=cam1_theta, radius=cam_r, arc=cam_arc, color=cam_c, lw=cam_lw)
geometry.eye_pov(ax=ax, xy=cam2_xy, angle=cam2_theta, radius=cam_r, arc=cam_arc, color=cam_c, lw=cam_lw)
geometry.eye_pov(ax=ax, xy=cam3_xy, angle=cam3_theta, radius=cam_r, arc=cam_arc, color=cam_c, lw=cam_lw)

target_r, target_l = humanoid_robot_2d.plot_robot(ax=ax, xy=robot_xy, q=q,
                                                  chest=chest, arms=arms, hands=hands, joints=joints, head=head)
plot_ray(cam_x=cam1_xy, target_x=target_r, color=ray_color_orientation, offset1=0.06, _ax=ax)
plot_ray(cam_x=cam1_xy, target_x=target_l, color=ray_color_orientation, offset1=0.06, _ax=ax)

plot_ray(cam_x=cam2_xy, target_x=target_r, color=ray_color_valid, offset1=0.04, _ax=ax)
plot_ray(cam_x=cam2_xy, target_x=target_l, color=ray_color_occlusion, offset1=0.17, _ax=ax)

plot_ray(cam_x=cam3_xy, target_x=target_r, color=ray_color_valid, offset1=0.04, _ax=ax)
plot_ray(cam_x=cam3_xy, target_x=target_l, color=ray_color_occlusion, offset1=0.205, _ax=ax)


# Legend
w, h = 1/golden_ratio**2, 1-1/golden_ratio
ax_legend = get_pip(ax=ax, x=0, y=0, width=w, height=h, aspect='auto',
                    xticks=[], yticks=[], frame_on=False)

y_legend = np.linspace(0.15, 0.85, 5)
pad = 0.1
offset = 0.02
# x0_l, x1_l, x2_l = 0.07*2*w, w*2/3, w*2/3+pad*0.5  # 2columns
x0_l, x1_l, x2_l = 0.05*2*w, w*3/5-0.02, w*3/5+pad*0.4
xd01 = x1_l - x0_l
ax_legend.set_xlim(0, 2*w)
ax_legend.set_ylim(0, 1)

ax_legend.add_patch(FancyBbox(xy=(offset*2*w, offset), width=2*w-2*offset*2*w, height=1-2*offset,
                              color='#f2f2f2', zorder=-10, boxstyle='Round', pad=pad))

# geometry.eye_pov(xy=(x0_l, y_legend[-1]), angle=0, radius=xd01/1.15, arc=cam_arc, ax=ax_legend, lw=cam_lw, color='k')
geometry.eye_pov(xy=(x0_l, y_legend[-1]), angle=0, radius=xd01/1.05, arc=cam_arc*1.35, ax=ax_legend,
                 lw=cam_lw*0.9, color='k')

plot_ray_legend(_ax=ax_legend, x=x0_l, y=y_legend[-2], dx=xd01, color=ray_color_valid)
plot_ray_legend(_ax=ax_legend, x=x0_l, y=y_legend[-3], dx=xd01, color=ray_color_occlusion)
plot_ray_legend(_ax=ax_legend, x=x0_l, y=y_legend[-4], dx=xd01, color=ray_color_orientation)

ww = xd01 * 0.7
eps = 0.01
ax_legend.add_patch(patches.FancyArrow(x=x0_l + xd01/2 + eps, y=y_legend[-5] - ww / 6, dx=0, dy=ww / 3,
                                       length_includes_head=True,
                                       overhang=-1, head_width=ww, head_length=ww / 6, color='k'))

ax_legend.text(x2_l, y_legend[-1], 'Camera System', ha=ha, va=va, size=font_size)
ax_legend.text(x2_l, y_legend[-2], 'Measurement', ha=ha, va=va, size=font_size)   # "Valid Measurement"
ax_legend.text(x2_l, y_legend[-3], 'Occlusion Error', ha=ha, va=va, size=font_size)
ax_legend.text(x2_l, y_legend[-4], 'Orientation Error', ha=ha, va=va, size=font_size)
ax_legend.text(x2_l, y_legend[-5], 'Target Marker', ha=ha, va=va, size=font_size)

save_fig(filename=ICHR20_CALIBRATION_FIGS+'Final/calibration_sketch', fig=fig, formats=('png', 'pdf'), bbox=None)

# fig.savefig(fig=fig, fname=ICHR20_CALIBRATION+'Final/Calibration_Sketch2.pdf')
