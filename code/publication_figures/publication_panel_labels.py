"""Place publication panel identifiers against a shared y-label reference.

Use the same final physical font size for single- and double-column figures.
Double-column canvases (177 mm) are inserted at the manuscript's 165.1 mm width.
"""


def add_panel_labels(fig, axes, labels, fontsize=12., insertion_scale=1.):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reference = next((a for a in axes if a.get_ylabel()), axes[0])
    if reference.get_ylabel():
        box = reference.yaxis.label.get_window_extent(renderer)
        offset = reference.bbox.x0 - (box.x0 + box.x1) / 2
    else:
        offset = 25. * fig.dpi / 72.
    artists = []
    for axis, text in zip(axes, labels):
        # Common physical offset, also for panels whose repeated y-label is hidden.
        x, y = fig.transFigure.inverted().transform(
            (axis.bbox.x0 - offset, axis.bbox.y1 + 2.*fig.dpi/72.))
        artist = fig.text(x, y, text, ha="center", va="bottom",
                          fontsize=fontsize/insertion_scale, family="Times New Roman")
        artist.set_gid("publication-panel-label")
        artists.append(artist)
    return artists
