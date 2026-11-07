import sys, xml.etree.ElementTree as E
def summary(p):
    r = E.parse(p).getroot()
    rc = sorted((x.get('name'), x.findtext('hardware/plugin')) for x in r.findall('ros2_control'))
    sens = sorted((g.get('reference'), se.findtext('update_rate'),
                   tuple(c.text for c in se.findall('contact/collision')), se.findtext('plugin/ros/remapping'))
                  for g in r.findall('gazebo') for se in g.findall('sensor') if se.get('type') == 'contact')
    return rc, sens, len(r.findall('link')), len(r.findall('joint'))
a, b = summary(sys.argv[1]), summary(sys.argv[2])
print('EQUIVALENT' if a == b else 'DIFFERENT:\n%s\n%s' % (a, b))
