//! Camera-coordinate vector plots for conditional motion fits, with sensitivity.
use serde_json::Value;
use std::fmt::Write as _;
type E = Box<dyn std::error::Error>;
fn vec3(v: &Value) -> Option<[f64; 3]> {
    Some([v[0].as_f64()?, v[1].as_f64()?, v[2].as_f64()?])
}
fn length(v: [f64; 3]) -> f64 {
    v.into_iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn project(v: [f64; 3], c: [f64; 2]) -> [f64; 2] {
    [
        c[0] + 78. * (0.866 * v[0] - 0.866 * v[2]),
        c[1] + 78. * (-0.5 * v[0] + v[1] - 0.5 * v[2]),
    ]
}
fn arrow(
    svg: &mut String,
    center: [f64; 2],
    vector: [f64; 3],
    color: &str,
    opacity: f64,
    dashed: bool,
) -> Result<(), E> {
    let n = length(vector);
    if n < 1e-10 {
        return Ok(());
    }
    let p = project(vector.map(|x| x / n), center);
    let d = [p[0] - center[0], p[1] - center[1]];
    let dn = d[0].hypot(d[1]).max(1e-10);
    let u = [d[0] / dn, d[1] / dn];
    write!(svg,"<g opacity='{opacity}' stroke='{color}' fill='{color}'><path d='M{},{} L{},{}' stroke-width='{}' stroke-dasharray='{}'/><path d='M{},{} L{},{} L{},{} Z' stroke='none'/></g>",center[0],center[1],p[0],p[1],if opacity<0.5 {1.2}else{3.},if dashed {"6 4"}else{"none"},p[0],p[1],p[0]-9.*u[0]+4.*u[1],p[1]-9.*u[1]-4.*u[0],p[0]-9.*u[0]-4.*u[1],p[1]-9.*u[1]+4.*u[0])?;
    Ok(())
}
fn axes(svg: &mut String, c: [f64; 2]) -> Result<(), E> {
    for (v, label) in [
        ([1., 0., 0.], "+x"),
        ([0., 1., 0.], "+y"),
        ([0., 0., 1.], "+z"),
    ] {
        let p = project(v, c);
        let q = project(v.map(|x| -0.9 * x), c);
        write!(svg,"<path d='M{},{} L{},{}' stroke='#56616e' stroke-width='1'/><text x='{}' y='{}' font-size='14' fill='#aab4c1'>{label}</text>",q[0],q[1],p[0],p[1],p[0]+5.,p[1]+5.)?;
    }
    write!(
        svg,
        "<circle cx='{}' cy='{}' r='3' fill='#bac4ce'/>",
        c[0], c[1]
    )?;
    Ok(())
}
fn formatted(v: &Value) -> String {
    vec3(v)
        .map(|v| format!("[{:+.3}, {:+.3}, {:+.3}]", v[0], v[1], v[2]))
        .unwrap_or_else(|| "unavailable".into())
}
pub fn panels(svg: &mut String, row: &Value) -> Result<(), E> {
    write!(svg,"<text x='30' y='887' font-size='19'>Pairwise 3D candidates relative to source {} · camera axes: x right, y down, z away</text>",row["source_sequence"])?;
    for (i, name, title, color) in [
        (0, "outer", "Outer top / bottom bands", "#ffb15c"),
        (1, "iris", "Iris region / reflection", "#62f4d3"),
    ] {
        let x = 30. + i as f64 * 785.;
        let set = &row[name];
        let base = &set["baseline"];
        let reference = row["reference"] == true;
        let sensitivity = set["sensitivity"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let probes = sensitivity
            .iter()
            .filter(|v| v["kind"] == "coordinate_perturbation")
            .collect::<Vec<_>>();
        let max_r = probes
            .iter()
            .filter_map(|v| v["rotation_difference_degrees"].as_f64())
            .fold(0., f64::max);
        let max_t = probes
            .iter()
            .filter_map(|v| v["translation_direction_difference_degrees"].as_f64())
            .fold(0., f64::max);
        let unavailable = base.is_null();
        let unstable =
            max_r > 2. || max_t > 20. || base["positive_depth_inliers"].as_u64().unwrap_or(0) < 8;
        let status = if reference {
            "REFERENCE"
        } else if unavailable {
            "UNAVAILABLE"
        } else if unstable {
            "UNSTABLE"
        } else {
            "CONDITIONAL FIT"
        };
        write!(svg,"<rect x='{x}' y='905' width='755' height='350' rx='15' fill='#202630'/><text x='{}' y='937' font-size='23' fill='{color}'>{title}</text><text x='{}' y='937' font-size='17' text-anchor='end' fill='#ffcb89'>{status}</text>",x+18.,x+737.)?;
        write!(svg,"<text x='{}' y='970' font-size='18'>Translation direction · unit length</text><text x='{}' y='970' font-size='18'>Rotation axis + angle</text>",x+22.,x+407.)?;
        for (k, field) in ["translation_unit_direction", "rotation_vector_degrees"]
            .iter()
            .enumerate()
        {
            let c = [x + 187. + k as f64 * 367., 1076.];
            axes(svg, c)?;
            for p in &probes {
                if let Some(v) = vec3(&p["fit"][*field]) {
                    arrow(svg, c, v, color, 0.19, false)?;
                }
            }
            if let Some(v) = vec3(&base[*field]) {
                arrow(svg, c, v, color, 1., unstable)?;
            }
        }
        if reference {
            write!(svg,"<text x='{}' y='1190' font-size='19'>Zero relative rotation; translation direction undefined at the reference.</text>",x+18.)?;
        } else if unavailable {
            write!(svg,"<text x='{}' y='1190' font-size='19'>Insufficient checked support for a 3D candidate.</text>",x+18.)?;
        } else {
            write!(svg,"<text x='{}' y='1182' font-family='monospace' font-size='16'>t {}</text><text x='{}' y='1182' font-family='monospace' font-size='16'>r {} deg</text>",x+18.,formatted(&base["translation_unit_direction"]),x+385.,formatted(&base["rotation_vector_degrees"]))?;
            write!(svg,"<text x='{}' y='1208' font-size='18'>Translation distance unknown</text><text x='{}' y='1208' font-size='18'>Rotation magnitude {:.3}°</text>",x+18.,x+385.,base["rotation_angle_degrees"].as_f64().unwrap_or(0.))?;
        }
        if !reference {
            let bound = if name == "outer" { 1. } else { 0.25 };
            write!(svg,"<text x='{}' y='1238' font-size='16'>{} points · ±{bound}px probes: direction Δ{max_t:.1}° / rotation Δ{max_r:.1}°</text>",x+18.,set["point_count"])?;
        } else {
            write!(svg,"<text x='{}' y='1238' font-size='16'>Arrows show directions; numbers retain rotation magnitude in degrees.</text>",x+18.)?;
        }
    }
    svg.push_str("<text x='30' y='1277' font-size='16' fill='#b8c0ca'>Faint arrows: coordinate-sensitivity probes, not probabilities. Dashed arrows: unstable fit. No metric 3D displacement is available.</text>");
    Ok(())
}
