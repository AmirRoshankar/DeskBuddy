---
name: DeskBuddy
description: Native controls for depth-guided SO-101 reaching
colors:
  canvas: "#171719"
  surface: "#202023"
  raised: "#29292D"
  border: "#3C3C42"
  text: "#F5F3F0"
  muted: "#B1AFAD"
  primary: "#246B49"
  primary-hover: "#1B5539"
  stop: "#AD3D35"
  stop-hover: "#8D302B"
  path: "#F46D67"
  focus: "#FF9275"
  disabled: "#344B3F"
  warning: "#F4B88C"
typography:
  title:
    fontFamily: Segoe UI
    fontSize: 10pt
    fontWeight: 700
  body:
    fontFamily: Segoe UI
    fontSize: 10pt
  label:
    fontFamily: Segoe UI
    fontSize: 9pt
  data:
    fontFamily: Consolas
    fontSize: 9pt
spacing:
  compact: 8px
  panel: 12px
  outer: 16px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#FFFFFF"
    padding: 9px 18px
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "#FFFFFF"
---

## Overview

A restrained desktop tool for someone operating a real arm at their desk.
The user's requested direction is dark grey, white, and minimal, with green Go,
red Stop and red planned paths; orange identifies joints and the height control.
The live scene leads; controls remain obvious and predictable. The native
application remains Tkinter, with no browser or additional UI runtime.

## Colors

`src/deskbuddy/ui/theme.py` is the implementation source. Go is green, Stop is
red, links are white, joint nodes are orange, and the planned path is red.
Depth keeps its existing blue-green-yellow measurement scale. Camera marker
colors retain their established meanings and written labels.

Body text and muted panel text retain their high-contrast white/grey treatment.

## Typography

Use Segoe UI for native controls and labels. Use Consolas only for event logs
and measurements. Do not turn operator instructions into decorative captions.

## Layout

Preserve the operator's requested arrangement: stacked zooms on the far left,
color and depth side by side, kinematics below color, activity below depth.
Hover height and actions occupy one row; connection/telemetry occupy the next.
There is no display title or redundant instructional footer. Logs retain detailed
events. Inspection boxes have fixed dimensions. Panels stretch with the window,
with 8 px gaps; camera content keeps its aspect ratio and click mapping through
letterboxing. Kinematics redraws at its actual panel size.

## Elevation & Depth

Flat tonal layers with one-pixel panel borders. No decorative shadows, glows,
gradients, or animated backgrounds.

## Shapes

Simple native rectangular controls. Keep consistent button padding and focus
outlines; visual state changes must not shift surrounding content.

## Components

`ui/widgets.py` owns buttons, panel headings, and the bounded event log.
`ui/layout.py` owns composition. These modules receive callbacks and do not
import motor transports or execute motion. Go stays bold white when disabled;
Stop retains hold; Recalibrate starts a new capture session.

Wheel and precision-trackpad input adjust preview hover height, with 2 mm per
full wheel notch and fractional increments supported. The continuous height track sits below the color image and eases
toward the new value. Replanning is debounced, and Go is disabled until the new
preview is valid. The nominal range is 2–20 cm above the selected surface; the
lower bound increases as needed for configured table clearance. Both bounds
intersect the same 3D workspace used by clicked targets. Scrolling over
Logs scrolls the log instead, and no height input directly commands motors.

## Do's and Don'ts

- Keep Go, Stop, and Recalibrate together at the top left.
- Show cached observations and stale frames as estimates.
- Preserve numeric units and separate requested from achieved control rates.
- Keep depth-map colors consistent between the main view and zoom.
- Do not change robot behavior as a side effect of visual state or startup.
- Do not present an encoder-projected tip as independent visual tracking.

The checked-in preview is an application-generated synthetic fixture, not a
desktop capture or evidence of a hardware run.

Camera overlays use white model links with orange joints, a red tip trajectory
with direction arrows, a dashed red planned end pose, and teal bounds at the
selected hover height with dashed risers to the paper. These are calibrated
model projections, not detected robot geometry or a swept collision envelope.

The header includes a mutually exclusive Numerical / Flow + IK selector. Switching
mode invalidates the old preview and never starts a reach. A blocked switch during
motor ownership restores the previously selected mode. Numerical remains the default.
