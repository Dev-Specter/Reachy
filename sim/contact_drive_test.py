import math, time, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState

LIDAR_X, BASE_R, SPEED, MAX_T = 0.184, 0.245, 0.10, 20.0
MASK = (math.radians(174), math.radians(186))
DEEP = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)

class T(Node):
    def __init__(s):
        super().__init__('contact_drive_test')
        s.pub = s.create_publisher(Twist, '/cmd_vel', 10)
        s.scan = s.odom = s.hit = None
        s.create_subscription(LaserScan, '/scan', lambda m: setattr(s, 'scan', m), qos_profile_sensor_data)
        s.create_subscription(Odometry, '/odom', lambda m: setattr(s, 'odom', m), qos_profile_sensor_data)
        for tp in ('/reachy/body_contact', '/reachy/arm_contact'):
            s.create_subscription(ContactsState, tp, lambda m, tp=tp: s.on_contact(tp, m), DEEP)
        s.t0, s.k, s.stop, s.done = time.time(), 0, 0, False
        s.create_timer(0.05, s.tick)

    def on_contact(s, tp, m):
        if m.states and s.hit is None: s.hit = (tp, m)

    def clear(s):
        m, best = s.scan, (float('inf'), None)
        for i, r in enumerate(m.ranges):
            a = m.angle_min + i * m.angle_increment
            if MASK[0] <= a <= MASK[1] or not (math.isfinite(r) and m.range_min < r < m.range_max): continue
            c = math.hypot(LIDAR_X + r * math.cos(a), r * math.sin(a)) - BASE_R
            if c < best[0]: best = (c, round(math.degrees(a)))
        return best

    def tick(s):
        stopping = s.hit is not None or time.time() - s.t0 >= MAX_T
        tw = Twist()
        if not stopping: tw.linear.x = SPEED
        s.pub.publish(tw)
        if stopping:
            s.stop += 1
            if s.stop == 1: s.report()
            if s.stop > 20: s.done = True
            return
        s.k += 1
        if s.k % 5 == 0 and s.scan and s.odom:
            c, a = s.clear()
            print(f"t={time.time()-s.t0:5.1f}s x={s.odom.pose.pose.position.x:6.3f} lidar_min_clear={c:6.3f} @ {a}deg", flush=True)

    def report(s):
        print("\n=== STOPPED:", "CONTACT on " + s.hit[0] if s.hit else "TIMEOUT (no contact)", "===")
        if s.hit:
            m = s.hit[1]
            print(f"  sensor frame: {m.header.frame_id}")
            for a, b in {(st.collision1_name, st.collision2_name) for st in m.states}: print(f"  {a}\n    <-> {b}")
        if s.scan: print(f"  lidar_min_clear at stop={s.clear()[0]:.3f}")
        if s.odom: print(f"  final x={s.odom.pose.pose.position.x:.3f}")

rclpy.init(); n = T()
while rclpy.ok() and not n.done: rclpy.spin_once(n, timeout_sec=0.05)
n.destroy_node(); rclpy.shutdown()
