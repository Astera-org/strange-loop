"""Generate an SVG with an n x n grid of unit-length arrows.

In row i, column j (both zero-indexed) the arrow makes an angle
    theta = i * j * 2*pi / n
with the positive x-axis (counter-clockwise, standard math convention).

Usage:
    python arrow_grid.py            # n=12 -> arrow_grid.svg
    python arrow_grid.py -n 16 -o out.svg --cell 50
"""
import math
import fire


def main(
    n, cell=60, margin=20, arrow_frac=0.8, stroke=2.0, outfile='arrow_grid.svg', rows=None):
    """Return an SVG document (str) for an n x n arrow grid."""
    size = n * cell + 2 * margin
    L = arrow_frac * cell          # on-screen length of a "unit" arrow
    half = L / 2.0

    # Arrowhead geometry (drawn via a reusable marker).
    head_len = 0.30 * L
    head_w = 0.22 * L

    out = []
    out.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            )
    out.append(f'  <rect width="{size}" height="{size}" fill="white"/>')

    # One marker, auto-oriented along each line, sized in user units.
    out.append('  <defs>')
    out.append(
            f'    <marker id="head" markerUnits="userSpaceOnUse" '
            f'markerWidth="{head_len}" markerHeight="{head_w}" '
            f'refX="{head_len}" refY="{head_w / 2}" orient="auto">'
            )
    out.append(
            f'      <path d="M0,0 L{head_len},{head_w / 2} L0,{head_w} Z" '
            f'fill="black"/>'
            )
    out.append('    </marker>')
    out.append('  </defs>')

    out.append(
            f'  <g stroke="black" stroke-width="{stroke}" '
            f'fill="none" marker-end="url(#head)">'
            )

    if rows is None:
        rows = range(n)

    for idx, i in enumerate(rows):
        for j in range(n):
            cx = margin + (j + 0.5) * cell
            cy = margin + (idx + 0.5) * cell
            theta = i * j * 2.0 * math.pi / n
            dx = math.cos(theta)
            dy = math.sin(theta)
            # SVG's y-axis points downward, so negate dy to keep the
            # angle counter-clockwise from the positive x-axis on screen.
            tail_x = cx - half * dx
            tail_y = cy + half * dy
            tip_x = cx + half * dx
            tip_y = cy - half * dy
            out.append(
                    f'    <line x1="{tail_x:.3f}" y1="{tail_y:.3f}" '
                    f'x2="{tip_x:.3f}" y2="{tip_y:.3f}"/>'
                    )

    out.append('  </g>')
    out.append('</svg>')

    with open(outfile, 'w') as f:
        f.write('\n'.join(out))
    print(f'Wrote {outfile}  (n={n})')


if __name__ == '__main__':
    fire.Fire(main)
