import math, time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState

LIDAR_X, BASE_R, SPEED, MAX_T = 0.184, 0.245, 0.15, 60.0
MASK = (math.radians(174), math.radians(186))   # mast self-hit

class WallTest(Node):
    def __init__(s):
        super().__init__('wall_test')
        s.pub = s.create_publisher(Twist, '/cmd_vel', 10)
        s.scan = s.odom = s.contact = None
        q = qos_profile_sensor_data
        s.create_subscription(LaserScan, '/scan', lambda m: setattr(s, 'scan', m), q)
        s.create_subscription(Odometry, '/odom', lambda m: setattr(s, 'odom', m), q)
        s.create_subscription(ContactsState, '/reachy/body_contact', s.on_contact, q)
        s.t0, s.k, s.stop_ticks, s.done = time.time(), 0, 0, False
        s.create_timer(0.05, s.tick)

    def on_contact(s, m):
        if m.states and s.contact is None:
            s.contact = m

    def clearance(s):
        m, best = s.scan, (float('inf'), None)
        for i, r in enumerate(m.ranges):
            a = m.angle_min + i * m.angle_increment
            if MASK[0] <= a <= MASK[1]: continue
            if not (math.isfinite(r) and m.range_min < r < m.range_max): continue
            c = math.hypot(LIDAR_X + r * math.cos(a), r * math.sin(a)) - BASE_R
            if c < best[0]: best = (c, math.degrees(a))
        return best

    def tick(s):
        stopping = s.contact is not None or time.time() - s.t0 >= MAX_T
        tw = Twist()
        if not stopping: tw.linear.x = SPEED
        s.pub.publish(tw)
        if stopping:
            s.stop_ticks += 1
            if s.stop_ticks == 1: s.report()
            if s.stop_ticks > 20: s.done = True
            return
        s.k += 1
        if not (s.scan and s.odom): return
        c, ang = s.clearance()
        front = s.scan.ranges[0]
        valid = math.isfinite(front) and s.scan.range_min < front < s.scan.range_max
        if (c < 1.0 and s.k % 5 == 0) or s.k % 40 == 0:
            print(f"t={time.time()-s.t0:5.1f}s x={s.odom.pose.pose.position.x:6.3f} "
                  f"front_raw={front:7.3f}{'' if valid else ' (INVALID)'} "
                  f"min_clear={c:6.3f} @ {ang if ang is None else round(ang)}deg", flush=True)

    def report(s):
        print("\n=== STOPPED:", "CONTACT" if s.contact else "TIMEOUT (no contact)", "===")
        if s.contact:
            pairs = {(st.collision1_name, st.collision2_name) for st in s.contact.states}
            for a, b in pairs: print(f"  {a}\n    <-> {b}")
            d = [x for st in s.contact.states for x in st.depths]
            if d: print(f"  max depth={max(d):.4f} m")
        if s.odom: print(f"  final odom x={s.odom.pose.pose.position.x:.3f} y={s.odom.pose.pose.position.y:.3f}")
        if s.scan: print(f"  front_raw at stop={s.scan.ranges[0]:.3f}")

rclpy.init(); n = WallTest()
while rclpy.ok() and not n.done: rclpy.spin_once(n, timeout_sec=0.05)
n.destroy_node(); rclpy.shutdown()
