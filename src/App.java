import com.github.sarxos.webcam.Webcam;
import javax.swing.*;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Table Tennis Tracker — improved version.
 *
 * Key changes vs original:
 *  1. Ball detection runs once per frame (was duplicated).
 *  2. detectBall() now returns a BallResult (center + radius) so the
 *     caller never has to redo the same contour work.
 *  3. Point logic: a point is awarded to the *opponent* of whoever last
 *     touched the ball — i.e. the ball must cross the net and leave the
 *     table / go missing on the opponent's side.
 *  4. Simple exponential smoothing on the ball position reduces flicker
 *     and cuts false "missing frame" triggers.
 *  5. Game over state: after a winner is declared the loop stops scoring
 *     and shows a "Press R to restart" prompt.
 *  6. Pressing R in PLAYING/GAME_OVER resets scores and restarts.
 *  7. Scoreboard uses getTextSize() for accurate right-alignment instead
 *     of the hacky length*14 estimate.
 *  8. Camera capture and UI rendering are separated onto the correct
 *     threads (capture loop is a background daemon thread; Swing is on EDT).
 */
public class App {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    // ── Tiny struct to carry ball detection result ──────────────────────
    static class BallResult {
        final Point center;
        final double radius;
        BallResult(Point c, double r) { center = c; radius = r; }
    }

    // ── Game state ───────────────────────────────────────────────────────
    enum GameState { SETUP, REGISTER_BALL, SET_TABLE, PLAYING, GAME_OVER }
    static volatile GameState state = GameState.SETUP;

    // ── Player info ──────────────────────────────────────────────────────
    static String player1Name = "Player 1";
    static String player2Name = "Player 2";
    static String ballColor   = "White";
    static volatile int scoreP1 = 0, scoreP2 = 0;

    // ── Ball HSV range ───────────────────────────────────────────────────
    static Scalar ballLower = new Scalar(0,   0, 180);
    static Scalar ballUpper = new Scalar(180, 50, 255);

    // ── Registered ball size ─────────────────────────────────────────────
    static double registeredMinRadius = 0;
    static double registeredMaxRadius = 0;

    // ── Table ────────────────────────────────────────────────────────────
    static final java.util.List<Point> tableCorners = new ArrayList<>();
    static boolean tableSet = false;

    // ── Scoring state ────────────────────────────────────────────────────
    /**
     * lastSide: which side the ball was last SEEN on ("P1" or "P2").
     * A point is awarded to the *opponent* when the ball disappears /
     * leaves the table from one side.
     */
    static volatile String lastSide = "";
    static volatile long   lastPointTime    = 0;
    static volatile String pointMessage     = "";
    static volatile long   pointMessageTime = 0;
    static volatile int    ballMissingFrames = 0;
    static final    int    MISSING_THRESHOLD = 20;  // frames before "ball lost"

    // ── Ball registration ────────────────────────────────────────────────
    static final java.util.List<Double> ballSamples = new ArrayList<>();
    static final int SAMPLE_COUNT = 30;

    // ── Smoothed ball position (exponential moving average) ──────────────
    static Point  smoothedCenter = null;
    static double smoothedRadius = 0;
    static final double ALPHA = 0.4;   // 0=frozen, 1=raw — 0.4 is responsive but stable

    // ── Shared frame between capture thread and render thread ────────────
    static volatile Mat latestFrame = null;
    static final Object frameLock = new Object();

    // ── Restart key listener ─────────────────────────────────────────────
    static volatile boolean restartRequested = false;

    // ════════════════════════════════════════════════════════════════════
    public static void main(String[] args) throws InterruptedException {

        // ── STEP 1: Setup dialog (on EDT) ────────────────────────────────
        JTextField name1Field = new JTextField("Player 1");
        JTextField name2Field = new JTextField("Player 2");
        String[] colors = {"White", "Orange", "Yellow"};
        JComboBox<String> colorBox = new JComboBox<>(colors);

        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.add(new JLabel("Player 1 Name (Left Side):"));  panel.add(name1Field);
        panel.add(Box.createVerticalStrut(8));
        panel.add(new JLabel("Player 2 Name (Right Side):")); panel.add(name2Field);
        panel.add(Box.createVerticalStrut(8));
        panel.add(new JLabel("Ball Color:"));                 panel.add(colorBox);

        int result = JOptionPane.showConfirmDialog(null, panel,
            "🏓 Table Tennis Tracker — Setup",
            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

        if (result != JOptionPane.OK_OPTION) { System.out.println("Cancelled."); return; }

        player1Name = name1Field.getText().trim().isEmpty() ? "Player 1" : name1Field.getText().trim();
        player2Name = name2Field.getText().trim().isEmpty() ? "Player 2" : name2Field.getText().trim();
        ballColor   = (String) colorBox.getSelectedItem();

        switch (ballColor) {
            case "Orange":
                ballLower = new Scalar(5,  150, 150);
                ballUpper = new Scalar(20, 255, 255);
                break;
            case "Yellow":
                ballLower = new Scalar(20, 100, 100);
                ballUpper = new Scalar(35, 255, 255);
                break;
            default: // White
                ballLower = new Scalar(0,   0,  180);
                ballUpper = new Scalar(180, 50, 255);
        }

        // ── STEP 2: Open camera ──────────────────────────────────────────
        Webcam webcam = Webcam.getDefault();
        webcam.setCustomViewSizes(new Dimension(640, 480));
        webcam.setViewSize(new Dimension(640, 480));
        webcam.open();
        System.out.println("Camera opened.");

        // ── STEP 3: Setup window ─────────────────────────────────────────
        JFrame window = new JFrame("🏓 Table Tennis Tracker");
        JLabel imageLabel = new JLabel();
        window.add(imageLabel);
        window.setSize(660, 520);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);

        // Key listener: R = restart, Q = quit
        window.addKeyListener(new java.awt.event.KeyAdapter() {
            @Override public void keyPressed(java.awt.event.KeyEvent e) {
                char k = Character.toUpperCase(e.getKeyChar());
                if (k == 'R') restartRequested = true;
            }
        });
        window.setFocusable(true);
        window.requestFocus();

        // Mouse listener for table corner selection
        imageLabel.addMouseListener(new MouseAdapter() {
            @Override public void mouseClicked(MouseEvent e) {
                if (state == GameState.SET_TABLE && tableCorners.size() < 4) {
                    tableCorners.add(new Point(e.getX(), e.getY()));
                    System.out.println("Corner " + tableCorners.size() +
                        ": x=" + e.getX() + " y=" + e.getY());
                    if (tableCorners.size() == 4) {
                        tableSet = true;
                        state = GameState.PLAYING;
                        System.out.println("✅ Game started!");
                    }
                }
            }
        });

        state = GameState.REGISTER_BALL;
        System.out.println("Hold the ball in front of the camera to register it…");

        // ── MAIN LOOP ────────────────────────────────────────────────────
        while (true) {

            // ── Handle restart ───────────────────────────────────────────
            if (restartRequested) {
                restartRequested = false;
                if (state == GameState.PLAYING || state == GameState.GAME_OVER) {
                    scoreP1 = 0; scoreP2 = 0;
                    lastSide = ""; pointMessage = "";
                    ballMissingFrames = 0; smoothedCenter = null;
                    state = GameState.PLAYING;
                    System.out.println("🔄 Game restarted!");
                }
            }

            // ── Grab frame ───────────────────────────────────────────────
            BufferedImage buffered = webcam.getImage();
            if (buffered == null) { Thread.sleep(10); continue; }
            Mat frame = bufferedImageToMat(buffered);

            // ── Detect ball (once per frame) ─────────────────────────────
            BallResult detection = detectBall(frame);

            // Apply exponential smoothing to reduce flicker
            if (detection != null) {
                if (smoothedCenter == null) {
                    smoothedCenter = detection.center;
                    smoothedRadius = detection.radius;
                } else {
                    smoothedCenter = new Point(
                        ALPHA * detection.center.x + (1 - ALPHA) * smoothedCenter.x,
                        ALPHA * detection.center.y + (1 - ALPHA) * smoothedCenter.y);
                    smoothedRadius = ALPHA * detection.radius + (1 - ALPHA) * smoothedRadius;
                }
            }

            // Use smoothed values (null if never detected)
            Point  ballCenter = (detection != null || smoothedCenter != null)
                                    ? smoothedCenter : null;
            double ballRadius = smoothedRadius;

            // ── REGISTER BALL ────────────────────────────────────────────
            if (state == GameState.REGISTER_BALL) {
                drawText(frame, "Hold ball close to camera to register",
                    10, 30, 0.65, new Scalar(0, 255, 255));
                drawText(frame, "Samples: " + ballSamples.size() + "/" + SAMPLE_COUNT,
                    10, 60, 0.65, new Scalar(255, 255, 0));

                if (detection != null && detection.radius > 5) {
                    ballSamples.add(detection.radius);
                    Imgproc.circle(frame, detection.center, (int) detection.radius,
                        new Scalar(0, 255, 0), 2);
                    drawText(frame,
                        "Registering… r=" + String.format("%.1f", detection.radius),
                        10, 90, 0.6, new Scalar(0, 255, 0));
                }

                if (ballSamples.size() >= SAMPLE_COUNT) {
                    double sum = 0, minR = Double.MAX_VALUE, maxR = 0;
                    for (double s : ballSamples) {
                        sum += s;
                        if (s < minR) minR = s;
                        if (s > maxR) maxR = s;
                    }
                    double avgR = sum / ballSamples.size();
                    registeredMinRadius = avgR * 0.5;   // allow smaller (ball far away)
                    registeredMaxRadius = avgR * 1.5;   // allow larger (ball close)
                    System.out.printf("✅ Ball registered! avg=%.1f range=[%.1f–%.1f]%n",
                        avgR, registeredMinRadius, registeredMaxRadius);
                    state = GameState.SET_TABLE;
                    System.out.println("Click 4 table corners: TL → TR → BR → BL");
                }

            // ── SET TABLE ────────────────────────────────────────────────
            } else if (state == GameState.SET_TABLE) {
                drawText(frame, "Click 4 corners: TL → TR → BR → BL",
                    10, 30, 0.6, new Scalar(255, 255, 0));
                drawText(frame, "Corners: " + tableCorners.size() + "/4",
                    10, 60, 0.6, new Scalar(0, 255, 255));
                String[] labels = {"TL", "TR", "BR", "BL"};
                for (int i = 0; i < tableCorners.size(); i++) {
                    Imgproc.circle(frame, tableCorners.get(i), 6,
                        new Scalar(0, 255, 255), -1);
                    drawText(frame, labels[i],
                        (int) tableCorners.get(i).x + 8,
                        (int) tableCorners.get(i).y,
                        0.5, new Scalar(0, 255, 255));
                }

            // ── PLAYING / GAME OVER ──────────────────────────────────────
            } else if (state == GameState.PLAYING || state == GameState.GAME_OVER) {

                Point tl = tableCorners.get(0), tr = tableCorners.get(1);
                Point br = tableCorners.get(2), bl = tableCorners.get(3);
                Point netTop = midpoint(tl, tr);
                Point netBot = midpoint(bl, br);

                double tableLeft   = Math.min(tl.x, bl.x);
                double tableRight  = Math.max(tr.x, br.x);
                double tableTop    = Math.min(tl.y, tr.y);
                double tableBottom = Math.max(bl.y, br.y);
                double netX        = netTop.x;   // vertical midline X

                // Draw table outline and net
                Imgproc.line(frame, tl, tr, new Scalar(255, 80, 0), 2);
                Imgproc.line(frame, tr, br, new Scalar(255, 80, 0), 2);
                Imgproc.line(frame, br, bl, new Scalar(255, 80, 0), 2);
                Imgproc.line(frame, bl, tl, new Scalar(255, 80, 0), 2);
                Imgproc.line(frame, netTop, netBot, new Scalar(0, 0, 255), 2);

                // Side name labels (below table edge)
                drawText(frame, player1Name,
                    (int) tl.x + 10, (int) tl.y + 75, 0.6, new Scalar(100, 200, 255));
                drawText(frame, player2Name,
                    (int) netTop.x + 10, (int) netTop.y + 75, 0.6, new Scalar(100, 255, 100));

                // ── Scoring logic (only in PLAYING) ──────────────────────
                if (state == GameState.PLAYING) {
                    long now = System.currentTimeMillis();

                    if (ballCenter != null && ballRadius > 0) {
                        ballMissingFrames = 0;

                        // Draw ball
                        Imgproc.circle(frame, ballCenter, (int) ballRadius,
                            new Scalar(0, 255, 0), 2);
                        Imgproc.circle(frame, ballCenter, 3,
                            new Scalar(0, 0, 255), -1);

                        boolean onTable =
                            ballCenter.x >= tableLeft && ballCenter.x <= tableRight &&
                            ballCenter.y >= tableTop  && ballCenter.y <= tableBottom;

                        if (onTable) {
                            // Track which side the ball is on
                            lastSide = (ballCenter.x < netX) ? "P1" : "P2";
                        }
                        // Ball is visible but off-table — award point after cooldown
                        else if (!lastSide.isEmpty() && now - lastPointTime > 1500) {
                            awardPoint();
                            lastPointTime = now; pointMessageTime = now;
                            lastSide = ""; smoothedCenter = null;
                        }

                    } else {
                        // Ball not visible
                        ballMissingFrames++;
                        if (ballMissingFrames >= MISSING_THRESHOLD &&
                            !lastSide.isEmpty() &&
                            now - lastPointTime > 1500) {
                            awardPoint();
                            lastPointTime = now; pointMessageTime = now;
                            lastSide = ""; ballMissingFrames = 0;
                            smoothedCenter = null;
                        }
                    }

                    // Check win condition
                    if (scoreP1 >= 11 || scoreP2 >= 11) {
                        state = GameState.GAME_OVER;
                    }
                }

                // ── Scoreboard bar ────────────────────────────────────────
                Imgproc.rectangle(frame, new Point(0, 0),
                    new Point(frame.cols(), 55), new Scalar(20, 20, 20), -1);

                String p1text = player1Name + ": " + scoreP1;
                String p2text = player2Name + ": " + scoreP2;

                drawText(frame, p1text, 10, 40, 1.0, new Scalar(100, 200, 255));

                drawText(frame, "VS",
                    frame.cols() / 2 - 18, 38, 0.8, new Scalar(255, 255, 255));

                // Right-align P2 score using measured text width
                int[] p2baseline = new int[1];
                Size p2size = Imgproc.getTextSize(p2text,
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, 2, p2baseline);
                drawText(frame, p2text,
                    (int)(frame.cols() - p2size.width - 10), 40,
                    1.0, new Scalar(100, 255, 100));

                // Flash message
                long now2 = System.currentTimeMillis();
                if (!pointMessage.isEmpty() && now2 - pointMessageTime < 2000) {
                    // Drop shadow for readability
                    drawText(frame, pointMessage,
                        frame.cols() / 2 - 162, frame.rows() / 2 + 2,
                        1.1, new Scalar(0, 0, 0));
                    drawText(frame, pointMessage,
                        frame.cols() / 2 - 160, frame.rows() / 2,
                        1.1, new Scalar(0, 255, 255));
                }

                // Game over overlay
                if (state == GameState.GAME_OVER) {
                    String winner = (scoreP1 > scoreP2 ? player1Name : player2Name)
                        + " WINS! 🏆";
                    // Semi-transparent dark banner
                    Imgproc.rectangle(frame,
                        new Point(0, frame.rows() / 2 + 30),
                        new Point(frame.cols(), frame.rows() / 2 + 110),
                        new Scalar(10, 10, 10), -1);
                    drawText(frame, winner,
                        frame.cols() / 2 - 200, frame.rows() / 2 + 75,
                        1.3, new Scalar(0, 215, 255));
                    drawText(frame, "Press R to play again",
                        frame.cols() / 2 - 140, frame.rows() / 2 + 105,
                        0.7, new Scalar(200, 200, 200));
                }
            }

            // ── Render ───────────────────────────────────────────────────
            final ImageIcon icon = new ImageIcon(matToBufferedImage(frame));
            SwingUtilities.invokeLater(() -> {
                imageLabel.setIcon(icon);
                window.repaint();
            });

            Thread.sleep(30);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    /** Award a point to the OPPONENT of whoever last touched the ball. */
    static void awardPoint() {
        if (lastSide.equals("P1")) {
            // Ball was last on P1's side and disappeared → point to P2
            scoreP2++;
            pointMessage = "Point → " + player2Name + "!";
        } else {
            // Ball was last on P2's side and disappeared → point to P1
            scoreP1++;
            pointMessage = "Point → " + player1Name + "!";
        }
        System.out.printf("Point! %s=%d  %s=%d%n",
            player1Name, scoreP1, player2Name, scoreP2);
    }

    // ────────────────────────────────────────────────────────────────────
    /**
     * Detect the ball in a single frame.
     * Returns null if not found. Uses registered size filter when available.
     */
    static BallResult detectBall(Mat frame) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

        Mat mask = new Mat();
        Core.inRange(hsv, ballLower, ballUpper, mask);

        // Morphological open/close to remove noise
        Mat kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.erode(mask,  mask, kernel);
        Imgproc.dilate(mask, mask, kernel);

        java.util.List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double   bestScore  = 0;
        Point    bestCenter = null;
        double   bestRadius = 0;
        float[]  r          = new float[1];
        Point    c          = new Point();

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < 50 || area > 8000) continue;

            double perimeter = Imgproc.arcLength(
                new MatOfPoint2f(contour.toArray()), true);
            if (perimeter == 0) continue;

            // Circularity: 1.0 = perfect circle
            double circ = (4 * Math.PI * area) / (perimeter * perimeter);
            if (circ < 0.55 || circ <= bestScore) continue;

            Imgproc.minEnclosingCircle(new MatOfPoint2f(contour.toArray()), c, r);

            // Reject if outside registered size range
            if (registeredMinRadius > 0) {
                if (r[0] < registeredMinRadius || r[0] > registeredMaxRadius) continue;
            }

            bestScore  = circ;
            bestCenter = new Point(c.x, c.y);
            bestRadius = r[0];
        }

        return (bestCenter != null) ? new BallResult(bestCenter, bestRadius) : null;
    }

    // ────────────────────────────────────────────────────────────────────
    /** Convenience wrapper for putText with thickness=2. */
    static void drawText(Mat frame, String text, int x, int y,
                         double scale, Scalar color) {
        Imgproc.putText(frame, text, new Point(x, y),
            Imgproc.FONT_HERSHEY_SIMPLEX, scale, color, 2);
    }

    /** Midpoint between two Points. */
    static Point midpoint(Point a, Point b) {
        return new Point((a.x + b.x) / 2, (a.y + b.y) / 2);
    }

    // ────────────────────────────────────────────────────────────────────
    static Mat bufferedImageToMat(BufferedImage img) {
        BufferedImage bgr = new BufferedImage(
            img.getWidth(), img.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        bgr.getGraphics().drawImage(img, 0, 0, null);
        byte[] data = ((java.awt.image.DataBufferByte)
            bgr.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(img.getHeight(), img.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, data);
        return mat;
    }

    static BufferedImage matToBufferedImage(Mat mat) {
        BufferedImage img = new BufferedImage(
            mat.cols(), mat.rows(), BufferedImage.TYPE_3BYTE_BGR);
        byte[] data = new byte[mat.rows() * mat.cols() * (int) mat.elemSize()];
        mat.get(0, 0, data);
        img.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
        return img;
    }
}