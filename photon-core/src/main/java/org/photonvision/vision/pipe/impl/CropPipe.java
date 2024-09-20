/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipe.impl;

import edu.wpi.first.math.MathUtil;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.photonvision.vision.pipe.CVPipe;

public class CropPipe extends CVPipe<Mat, Mat, Rect> {
    public CropPipe() {
        this.params = new Rect(0, 0, Integer.MAX_VALUE, Integer.MAX_VALUE);
    }

    @Override
    protected Mat process(Mat in) {
        if (this.fullyCovers(in)) {
            return in;
        }

        int x = MathUtil.clamp(params.x, 0, in.width());
        int y = MathUtil.clamp(params.y, 0, in.height());
        int width = MathUtil.clamp(params.width, 0, in.width() - x);
        int height = MathUtil.clamp(params.height, 0, in.height() - y);

        return in.submat(y, y + height, x, x + width);
    }

    public static Rect intersection(Rect a, Rect b) {
        int x = Math.max(a.x, b.x);
        int y = Math.max(a.y, b.y);
        int width = Math.min(a.x + a.width, b.x + b.width) - x;
        int height = Math.min(a.y + a.height, b.y + b.height) - y;

        if (width <= 0 || height <= 0) {
            return new Rect();
        }

        return new Rect(x, y, width, height);
    }

    private boolean fullyCovers(int imageWidth, int imageHeight) {
        return params.x == 0
                && params.y == 0
                && params.width == imageWidth
                && params.height == imageHeight;
    }

    private boolean fullyCovers(Mat mat) {
        return fullyCovers(mat.width(), mat.height());
    }
}
