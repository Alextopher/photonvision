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

package org.photonvision.vision.opencv;

public enum ContourShape {
    Circle(0),
    Custom(-1),
    Triangle(3),
    Quadrilateral(4);

    public final int sides;

    ContourShape(int sides) {
        this.sides = sides;
    }

    public static ContourShape fromSides(int sides) {
        switch (sides) {
            case 0:
                return Circle;
            case 3:
                return Triangle;
            case 4:
                return Quadrilateral;
            default:
                return Custom;
        }
    }
}
