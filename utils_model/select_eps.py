import numpy as np
import cv2
from matplotlib import pyplot as plt


class DemoSelectEps_new:
    def __init__(
        self,
        target_mask,
        mid_weight,
        orther_weight,
        pattern_name,
    ):
        self.target = target_mask
        self.mid_weight = mid_weight
        self.orther_weight = orther_weight
        self.pattern_name = pattern_name

    def fix_cps(self, ls_mask, k, symmetry="left-right"):
        """
        Extract evenly spaced contour control points and build a symmetric partner set.
        """
        mask_uint8 = (ls_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_simpling_mcps = []

        for contour in contours:
            contour_pts = contour[:, 0, [1, 0]].astype(np.float32)
            sampling_pts = self.sample_elements(contour_pts, k, "skip")
            all_simpling_mcps.append(sampling_pts)

        center = [ls_mask.shape[0] // 2, ls_mask.shape[1] // 2]

        if len(all_simpling_mcps) > 1:
            if symmetry == "center":
                all_simpling_mcps[1] = [
                    [2 * center[0] - y, 2 * center[1] - x] for y, x in all_simpling_mcps[0]
                ]
            elif symmetry == "left-right":
                all_simpling_mcps[1] = [
                    [y, 2 * center[1] - x] for y, x in all_simpling_mcps[0]
                ]
            elif symmetry == "diagonal":
                h = ls_mask.shape[0]
                w = ls_mask.shape[1]
                all_simpling_mcps[1] = [
                    [h - 1 - x, w - 1 - y] for y, x in all_simpling_mcps[0]
                ]
            else:
                raise ValueError("symmetry must be one of: center, left-right, diagonal")

        return all_simpling_mcps

    def fix_OPC_cps(self, ls_mask, k):
        mask_uint8 = (ls_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_simpling_mcps = []

        for contour in contours:
            contour_pts = contour[:, 0, [1, 0]].astype(np.float32)
            sampling_pts = self.sample_elements(contour_pts, k, "skip")
            all_simpling_mcps.append(sampling_pts)

        return all_simpling_mcps

    def _select_eps_ofothers(self, interval_line: int = 6, interval_corner: int = 0):
        """
        Sample EPE evaluation points on the contour.

        Current behavior:
        - Segment endpoints keep an `interval_corner` exclusion distance.
        - The endpoint-side protection points are always kept active.
        - The interior keeps about 70% of sampled candidates.
        - Very short diagonal corner-connectors from pixelized 90-degree corners are filtered.
        """
        mask_uint8 = (self.target * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_evaluation_points = []
        core_region_points = []

        for contour in contours:
            pts = contour[:, 0, :]
            pts = np.column_stack((pts[:, 1], pts[:, 0]))
            num_pts = len(pts)

            for i in range(num_pts):
                start_pt = pts[i]
                end_pt = pts[(i + 1) % num_pts]
                line_coords = self._bresenham_line_with_startandend(*start_pt, *end_pt)
                seg_len = len(line_coords)
                if seg_len < 1:
                    continue

                left_limit = min(max(0, interval_corner), seg_len - 1)
                right_limit = max(
                    0,
                    seg_len - 1 - min(max(0, interval_corner), seg_len - 1),
                )

                if self._is_short_corner_connector(pts, i, seg_len, left_limit, right_limit):
                    continue

                mid_idx = seg_len // 2
                if left_limit > right_limit:
                    left_limit = right_limit = mid_idx

                indices_set = {mid_idx}
                for idx in range(mid_idx - interval_line, left_limit - 1, -interval_line):
                    indices_set.add(idx)
                for idx in range(mid_idx + interval_line, right_limit + 1, interval_line):
                    indices_set.add(idx)
                if left_limit < seg_len:
                    indices_set.add(left_limit)
                if right_limit >= 0:
                    indices_set.add(right_limit)

                sampled_indices = sorted(idx for idx in indices_set if 0 <= idx < seg_len)
                if not sampled_indices:
                    continue

                current_segment_samples = [line_coords[idx] for idx in sampled_indices]
                all_evaluation_points.extend(current_segment_samples)

                left_mid_gap = mid_idx - sampled_indices[0]
                right_mid_gap = sampled_indices[-1] - mid_idx

                if (
                    sampled_indices[0] != sampled_indices[-1]
                    and left_mid_gap < interval_line
                    and right_mid_gap < interval_line
                ):
                    final_indices = [sampled_indices[0], sampled_indices[-1]]
                elif len(sampled_indices) <= 2:
                    final_indices = list(sampled_indices)
                else:
                    interior_indices = sampled_indices[1:-1]
                    if len(interior_indices) == 0:
                        final_indices = [sampled_indices[0], sampled_indices[-1]]
                    else:
                        target_interior_count = int(round(len(interior_indices) * 0.7))
                        target_interior_count = max(
                            1,
                            min(len(interior_indices), target_interior_count),
                        )
                        if target_interior_count % 2 == 0:
                            if target_interior_count < len(interior_indices):
                                target_interior_count += 1
                            else:
                                target_interior_count -= 1

                        if target_interior_count <= 0:
                            selected_interior = []
                        else:
                            try:
                                mid_pos_in_list = sampled_indices.index(mid_idx)
                            except ValueError:
                                mid_pos_in_list = min(
                                    range(len(sampled_indices)),
                                    key=lambda k: abs(sampled_indices[k] - mid_idx),
                                )
                            mid_pos_interior = max(
                                0,
                                min(len(interior_indices) - 1, mid_pos_in_list - 1),
                            )
                            radius = (target_interior_count - 1) // 2
                            start_k = max(0, mid_pos_interior - radius)
                            end_k = min(len(interior_indices), start_k + target_interior_count)
                            start_k = max(0, end_k - target_interior_count)
                            selected_interior = interior_indices[start_k:end_k]

                        final_indices = (
                            [sampled_indices[0]]
                            + list(selected_interior)
                            + [sampled_indices[-1]]
                        )
                        final_indices = list(dict.fromkeys(final_indices))

                core_region_points.extend([line_coords[idx] for idx in final_indices])

        initial_eps = np.array(all_evaluation_points)
        if len(initial_eps) == 0:
            return np.empty((0, 2)), np.empty((1, 0)), np.empty((1, 0))

        core_points_set = set(map(tuple, core_region_points))
        num_eps = len(initial_eps)
        weight_meef = np.ones(num_eps)
        is_core = np.array([tuple(pt) in core_points_set for pt in initial_eps])
        weight_meef[is_core] = self.mid_weight
        weight_meef = weight_meef.reshape(1, -1)

        weight_epe = np.zeros_like(weight_meef)
        weight_epe[weight_meef == self.mid_weight] = 1.0
        return initial_eps, weight_epe, weight_meef

    def _segment_orientation(self, start_pt, end_pt):
        dy = int(end_pt[0]) - int(start_pt[0])
        dx = int(end_pt[1]) - int(start_pt[1])
        if dy == 0 and dx != 0:
            return "horizontal"
        if dx == 0 and dy != 0:
            return "vertical"
        if dx != 0 and dy != 0:
            return "diagonal"
        return "point"

    def _is_short_corner_connector(self, pts, seg_idx, seg_len, left_limit, right_limit):
        """
        Filter short diagonal connector segments introduced by pixelized 90-degree corners.

        Typical pattern:
        - previous segment: vertical
        - current segment: short diagonal connector
        - next segment: horizontal

        Once the interval-corner protection zones overlap on that diagonal connector,
        keeping points there tends to create false EPs at the corner itself.
        """
        if seg_len < 1:
            return False

        current_start = pts[seg_idx]
        current_end = pts[(seg_idx + 1) % len(pts)]
        if self._segment_orientation(current_start, current_end) != "diagonal":
            return False

        prev_start = pts[(seg_idx - 1) % len(pts)]
        prev_end = pts[seg_idx]
        next_start = pts[(seg_idx + 1) % len(pts)]
        next_end = pts[(seg_idx + 2) % len(pts)]
        prev_orientation = self._segment_orientation(prev_start, prev_end)
        next_orientation = self._segment_orientation(next_start, next_end)

        if {prev_orientation, next_orientation} != {"horizontal", "vertical"}:
            return False

        return left_limit > right_limit

    def debug_plot_eps(
        self,
        eps_pts,
        weight_epe=None,
        weight_meef=None,
        save_path=None,
        show=True,
        core_offset=None,
        core_size=None,
    ):
        """
        Plot sampled EP points on top of the target mask.

        只显示 weight_epe 中权重大于 0 的点 (即真正参与 wEPE 计算的位置),
        并以纯红色绘制. 保存时不带颜色条与标题.
        """
        img = self.target.astype(float)
        pts = np.asarray(eps_pts).reshape(-1, 2)

        # 仅保留 weight_epe > 0 的点; 若未提供 weight_epe 则全部保留
        if weight_epe is not None:
            we = np.asarray(weight_epe).reshape(-1)
            mask_keep = we > 0
            pts = pts[mask_keep]

        ys = pts[:, 0]
        xs = pts[:, 1]

        # 用 add_axes((0,0,1,1)) 让坐标系完全填满画布,
        # 配合 savefig 的 pad_inches=0 彻底去掉白边.
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.scatter(xs, ys, s=20, marker="o", c="red")

        if core_offset is not None and core_size is not None:
            gy, gx = core_offset
            ch, cw = core_size
            import matplotlib.patches as patches

            rect = patches.Rectangle(
                (gx, gy),
                cw,
                ch,
                linewidth=3,
                edgecolor="yellow",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_axis_off()
        # 锁定坐标范围 = 图像像素范围, 不让 scatter 自动扩边
        ax.set_xlim(-0.5, img.shape[1] - 0.5)
        ax.set_ylim(img.shape[0] - 0.5, -0.5)  # imshow 默认 y 轴向下
        plt.savefig(F"{save_path}/debug_eps.png", dpi=300,
                    bbox_inches="tight", pad_inches=0)
        if show:
            plt.show()
        plt.close(fig)

    def extract_mask_control_points(self, k: int, symmetry=None):
        """
        Unified contour control-point extraction for normal and symmetry-constrained cases.
        """
        mask_uint8 = (self.target * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        all_sampling_mcps = []
        for contour in contours:
            contour_pts = contour[:, 0, [1, 0]].astype(np.float32)
            sampling_pts = self.sample_elements(contour_pts, k, "skip")
            all_sampling_mcps.append(sampling_pts)

        if symmetry is None or len(all_sampling_mcps) < 2:
            return all_sampling_mcps

        h, w = self.target.shape
        center_y, center_x = h // 2, w // 2
        ref_pts = all_sampling_mcps[0]

        if symmetry == "center":
            new_pts = [[2 * center_y - y, 2 * center_x - x] for y, x in ref_pts]
        elif symmetry == "left-right":
            new_pts = [[y, 2 * center_x - x] for y, x in ref_pts]
        elif symmetry == "diagonal":
            new_pts = [[h - 1 - x, w - 1 - y] for y, x in ref_pts]
        else:
            raise ValueError(
                f"Unsupported symmetry type: {symmetry}. "
                "Choose center, left-right, diagonal, or None."
            )

        all_sampling_mcps[1] = new_pts
        return all_sampling_mcps

    def sample_elements(self, lst, k, mode="step"):
        """
        Sample every k-th element or every (k+1)-th element from a list-like input.
        """
        if mode == "step":
            return np.array(lst[::k])
        if mode == "skip":
            return np.array(lst[:: k + 1])
        raise ValueError("mode must be 'step' or 'skip'")

    def _bresenham_line_with_startandend(self, y1, x1, y2, x2):
        points = [(y1, x1)]
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        sy = 1 if y2 > y1 else -1
        sx = 1 if x2 > x1 else -1
        err = dx - dy

        while (y1, x1) != (y2, x2):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
            points.append([y1, x1])

        return points

    def remove_adjacent_close_points(self, nested_list, threshold=2):
        """
        Remove adjacent contour points whose spacing is below the given threshold.
        """
        cleaned_contours = []

        for contour in nested_list:
            n = len(contour)
            if n == 0:
                cleaned_contours.append([])
                continue
            if n == 1:
                cleaned_contours.append(contour[:])
                continue

            result = []
            for i in range(n):
                prev_idx = (i - 1) % n
                dist = np.linalg.norm(np.array(contour[i]) - np.array(contour[prev_idx]))
                if dist >= threshold:
                    result.append(contour[i])
            cleaned_contours.append(result)

        return cleaned_contours

    def _select_eps_ofvia(self, r: int = 2):
        mask_uint8 = (self.target * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cps = []
        wepe_caculated = []
        weight_meef = []

        for contour in contours:
            contour_pts = [[pt[0][1], pt[0][0]] for pt in contour]
            length = len(contour_pts)
            sample_coords = []
            for i in range(length):
                start = contour_pts[i]
                end = contour_pts[(i + 1) % length]
                coords = self._bresenham_line_with_startandend(*start, *end)

                if len(coords) < 4:
                    continue

                if i < 4:
                    corner_offset = [[r, r], [-r, r], [-r, -r], [r, -r]][i]
                    cy, cx = coords[0][0] + corner_offset[0], coords[0][1] + corner_offset[1]
                    sample_coords.append([cy, cx])
                    wepe_caculated.extend([0])
                    weight_meef.extend([self.mid_weight])

                points_to_add = [
                    coords[len(coords) // 4],
                    coords[len(coords) // 3],
                    coords[len(coords) // 2],
                    coords[-1 - len(coords) // 3],
                    coords[-1 - len(coords) // 4],
                ]

                wepe_caculated.extend([0])
                wepe_caculated.extend([0])
                weight_meef.extend([self.orther_weight])
                weight_meef.extend([self.orther_weight])
                wepe_caculated.extend([1])
                weight_meef.extend([self.mid_weight])
                wepe_caculated.extend([0])
                weight_meef.extend([self.orther_weight])
                wepe_caculated.extend([0])
                weight_meef.extend([self.orther_weight])

                for pt in points_to_add:
                    py, px = pt
                    sample_coords.append([py, px])

            cps.append(sample_coords)
            initial_eps = [pt for contour in cps for pt in contour]
            initial_eps = np.asarray(initial_eps).reshape(-1, 2)

        return initial_eps, wepe_caculated, weight_meef
