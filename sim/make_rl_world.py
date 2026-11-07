import sys, xml.etree.ElementTree as E
src, dst = sys.argv[1], sys.argv[2]
t = E.parse(src); w = t.getroot().find('world')
p = w.find('physics')
print('existing physics:', None if p is None else E.tostring(p).decode().strip())
if p is None:
    p = E.SubElement(w, 'physics', {'name': 'default_physics', 'default': '1', 'type': 'ode'})
for tag, val in (('max_step_size', '0.001'), ('real_time_factor', '1'), ('real_time_update_rate', '0')):
    e = p.find(tag)
    if e is None: e = E.SubElement(p, tag)
    e.text = val
if not any(pl.get('filename') == 'libgazebo_ros_state.so' for pl in w.findall('plugin')):
    pl = E.SubElement(w, 'plugin', {'name': 'gazebo_ros_state', 'filename': 'libgazebo_ros_state.so'})
    ros = E.SubElement(pl, 'ros'); E.SubElement(ros, 'namespace').text = '/'
    E.SubElement(pl, 'update_rate').text = '1.0'
t.write(dst, xml_declaration=True, encoding='utf-8')
print('written', dst)
