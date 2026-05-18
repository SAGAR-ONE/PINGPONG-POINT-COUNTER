import com.github.sarxos.webcam.Webcam;
import javax.swing.*;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Table Tennis Tracker — Full Rewrite
 *
 * New features added:
 *  1. Trajectory Prediction  — stores last N ball positions, extrapolates
 *                              next position when ball is briefly lost.
 *  2. Bounce Detection       — tracks vertical velocity sign changes;
 *                              two bounces on the same side = point.
 *  3. Perspective Correction — getPerspectiveTransform (homography) maps
 *                              the trapezoid camera view to a flat rectangle
 *                              so all geometry is accurate.
 *  4. Rally Counter + Stats  — counts net crossings per rally; after game
 *                              over shows a full stats screen.
 *
 * Retained from previous version:
 *  - Ball registration (30-frame sampling)
 *  - HSV color selection (White / Orange / Yellow)
 *  - Exponential smoothing on ball position
 *  - Cooldown between points
 *  - Game over + Press R to restart
 *  - Right-aligned scoreboard using getTextSize()
 */
public class App {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    // ── Tiny structs ─────────────────────────────────────────────────────

    static class BallResult {
        final Point  center;
        final double radius;
        BallResult(Point c, double r) { center = c; radius = r; }
    }

    static class MatchStats {
        int   totalPointsPlayed  = 0;
        int   longestRally       = 0;
        int   currentRally       = 0;
        int   p1MaxConsecutive   = 0;
        int   p2MaxConsecutive   = 0;
        int   p1Consecutive      = 0;
        int   p2Consecutive      = 0;
        long  totalRallyFrames   = 0;  // for average rally length estimate
        int   rallyCount         = 0;

        void recordPoint(String scorer) {
            totalPointsPlayed++;
            if (currentRally > longestRally) longestRally = currentRally;
            totalRallyFrames += currentRally;
            rallyCount++;
            currentRally = 0;

            if (scorer.equals("P1")) {
                p1Consecutive++;
                p2Consecutive = 0;
                if (p1Consecutive > p1MaxConsecutive) p1MaxConsecutive = p1Consecutive;
            } else {
                p2Consecutive++;
                p1Consecutive = 0;
                if (p2Consecutive > p2MaxConsecutive) p2MaxConsecutive = p2Consecutive;
            }
        }

        void netCrossing() { currentRally++; }

        double avgRallyLength() {
            return rallyCount == 0 ? 0 : (double) totalRallyFrames / rallyCount;
        }
    }

    // ── Game state ────────────────────────────────────────────────────────
    enum GameState { SETUP, REGISTER_BALL, SET_TABLE, PLAYING, GAME_OVER }
    static volatile GameState state = GameState.SETUP;

    // ── Player info ───────────────────────────────────────────────────────
    static String player1Name = "Player 1";
    static String player2Name = "Player 2";
    static String ballColor   = "White";
    static volatile int scoreP1 = 0, scoreP2 = 0;

    // ── Ball HSV range ────────────────────────────────────────────────────
    static Scalar ballLower = new Scalar(0,   0, 180);
    static Scalar ballUpper = new Scalar(180, 50, 255);

    // ── Registered ball size ──────────────────────────────────────────────
    static double registeredMinRadius = 0;
    static double registeredMaxRadius = 0;
    static final List<Double> ballSamples = new ArrayList<>();
    static final int SAMPLE_COUNT = 30;

    // ── Table corners (raw camera) ────────────────────────────────────────
    static final List<Point> tableCorners = new ArrayList<>();
    static boolean tableSet = false;

    // ── Homography ────────────────────────────────────────────────────────
    // Maps camera frame → flat top-down table rectangle (640 × 320 px)
    static Mat homographyMatrix   = null;
    static Mat homographyInverse  = null;
    static final int FLAT_W = 640;
    static final int FLAT_H = 320;

    // ── Trajectory Prediction ─────────────────────────────────────────────
    static final int TRAJ_HISTORY = 8;
    static final LinkedList<Point> posHistory = new LinkedList<>();  // in FLAT space
    static Point predictedPos = null;   // last extrapolated position (flat space)

    // ── Bounce Detection ──────────────────────────────────────────────────
    // We track vertical position in FLAT space. A sign change in dy = bounce.
    static double prevFlatY     = -1;
    static double prevDY        = 0;
    static String bounceSide    = "";   // "P1" or "P2"
    static int    bounceCount   = 0;    // consecutive bounces on same side
    static final int BOUNCE_THRESHOLD = 2;  // bounces to award point

    // ── Scoring state ─────────────────────────────────────────────────────
    static volatile String lastSide          = "";
    static volatile long   lastPointTime     = 0;
    static volatile String pointMessage      = "";
    static volatile long   pointMessageTime  = 0;
    static volatile int    ballMissingFrames = 0;
    static final    int    MISSING_THRESHOLD = 25;

    // Was the ball on P1's side last frame? Used for net-crossing detection.
    static boolean wasOnP1Side = false;
    static boolean ballSeenOnce = false;  // ignore first crossing noise

    // ── Smoothed ball position (raw camera space) ─────────────────────────
    static Point  smoothedCenter = null;
    static double smoothedRadius = 0;
    static final double ALPHA = 0.4;

    // ── Stats ─────────────────────────────────────────────────────────────
    static final MatchStats stats = new MatchStats();

    // ── Restart flag ──────────────────────────────────────────────────────
    static volatile boolean restartRequested = false;

    // ═════════════════════════════════════════════════════════════════════
    public static void main(String[] args) throws InterruptedException {

        // ── Setup dialog ─────────────────────────────────────────────────
        JTextField name1Field = new JTextField("Player 1");
        JTextField name2Field = new JTextField("Player 2");
        String[]   colors     = {"White", "Orange", "Yellow"};
        JComboBox<String> colorBox = new JComboBox<>(colors);

        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.add(new JLabel("Player 1 Name (Left Side):"));  panel.add(name1Field);
        panel.add(Box.createVerticalStrut(8));
        panel.add(new JLabel("Player 2 Name (Right Side):")); panel.add(name2Field);
        panel.add(Box.createVerticalStrut(8));
        panel.add(new JLabel("Ball Color:"));                 panel.add(colorBox);

        int dlgResult = JOptionPane.showConfirmDialog(null, panel,
            "🏓 Table Tennis Tracker — Setup",
            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
        if (dlgResult != JOptionPane.OK_OPTION) { System.out.println("Cancelled."); return; }

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

        // ── Open camera ──────────────────────────────────────────────────
        Webcam webcam = Webcam.getDefault();
        webcam.setCustomViewSizes(new Dimension(640, 480));
        webcam.setViewSize(new Dimension(640, 480));
        webcam.open();
        System.out.println("Camera opened.");

        // ── Build window ──────────────────────────────────────────────────
        JFrame window = new JFrame("🏓 Table Tennis Tracker");
        JLabel imageLabel = new JLabel();
        window.add(imageLabel);
        window.setSize(660, 520);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);

        window.addKeyListener(new java.awt.event.KeyAdapter() {
            @Override public void keyPressed(java.awt.event.KeyEvent e) {
                if (Character.toUpperCase(e.getKeyChar()) == 'R') restartRequested = true;
            }
        });
        window.setFocusable(true);
        window.requestFocus();

        imageLabel.addMouseListener(new MouseAdapter() {
            @Override public void mouseClicked(MouseEvent e) {
                if (state == GameState.SET_TABLE && tableCorners.size() < 4) {
                    tableCorners.add(new Point(e.getX(), e.getY()));
                    System.out.printf("Corner %d: x=%d y=%d%n",
                        tableCorners.size(), e.getX(), e.getY());
                    if (tableCorners.size() == 4) {
                        buildHomography();
                        tableSet = true;
                        state = GameState.PLAYING;
                        System.out.println("✅ Homography built. Game started!");
                    }
                }
            }
        });

        state = GameState.REGISTER_BALL;
        System.out.println("Hold the ball in front of the camera to register…");

        // ── MAIN LOOP ─────────────────────────────────────────────────────
        while (true) {

            // Handle restart
            if (restartRequested) {
                restartRequested = false;
                if (state == GameState.PLAYING || state == GameState.GAME_OVER) {
                    resetGame();
                }
            }

            BufferedImage buffered = webcam.getImage();
            if (buffered == null) { Thread.sleep(10); continue; }
            Mat frame = bufferedImageToMat(buffered);

            // Detect ball in raw camera frame
            BallResult detection = detectBall(frame);

            // Smooth in raw camera space
            if (detection != null) {
                smoothedCenter = (smoothedCenter == null) ? detection.center
                    : new Point(
                        ALPHA * detection.center.x + (1 - ALPHA) * smoothedCenter.x,
                        ALPHA * detection.center.y + (1 - ALPHA) * smoothedCenter.y);
                smoothedRadius = (smoothedCenter == null) ? detection.radius
                    : ALPHA * detection.radius + (1 - ALPHA) * smoothedRadius;
            }

            // ── REGISTER BALL ─────────────────────────────────────────────
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
                    registeredMinRadius = avgR * 0.5;
                    registeredMaxRadius = avgR * 1.5;
                    System.out.printf("✅ Ball registered! avg=%.1f range=[%.1f–%.1f]%n",
                        avgR, registeredMinRadius, registeredMaxRadius);
                    state = GameState.SET_TABLE;
                    System.out.println("Click 4 corners: TL → TR → BR → BL");
                }

            // ── SET TABLE ─────────────────────────────────────────────────
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

            // ── PLAYING / GAME OVER ───────────────────────────────────────
            } else if (state == GameState.PLAYING || state == GameState.GAME_OVER) {

                // Draw table outline in raw camera space (for visual reference)
                Point tl = tableCorners.get(0), tr = tableCorners.get(1);
                Point br = tableCorners.get(2), bl = tableCorners.get(3);
                Imgproc.line(frame, tl, tr, new Scalar(255, 80, 0), 2);
                Imgproc.line(frame, tr, br, new Scalar(255, 80, 0), 2);
                Imgproc.line(frame, br, bl, new Scalar(255, 80, 0), 2);
                Imgproc.line(frame, bl, tl, new Scalar(255, 80, 0), 2);

                // Draw net (midpoint line in raw space)
                Point netTop = midpoint(tl, tr);
                Point netBot = midpoint(bl, br);
                Imgproc.line(frame, netTop, netBot, new Scalar(0, 0, 255), 2);

                // Player labels
                drawText(frame, player1Name,
                    (int) tl.x + 10, (int) tl.y + 75, 0.6, new Scalar(100, 200, 255));
                drawText(frame, player2Name,
                    (int) netTop.x + 10, (int) netTop.y + 75, 0.6, new Scalar(100, 255, 100));

                // ── Scoring logic (PLAYING only) ──────────────────────────
                if (state == GameState.PLAYING) {
                    long now = System.currentTimeMillis();

                    // Get ball position in FLAT (homography-corrected) space
                    Point flatBall = null;
                    if (smoothedCenter != null) {
                        flatBall = toFlatSpace(smoothedCenter);
                    } else if (!posHistory.isEmpty()) {
                        // Use trajectory prediction if ball is briefly lost
                        flatBall = predictNextPosition();
                        predictedPos = flatBall;
                    }

                    if (smoothedCenter != null && detection != null) {
                        ballMissingFrames = 0;
                        predictedPos = null;

                        // Draw ball on raw frame
                        Imgproc.circle(frame, smoothedCenter, (int) smoothedRadius,
                            new Scalar(0, 255, 0), 2);
                        Imgproc.circle(frame, smoothedCenter, 3,
                            new Scalar(0, 0, 255), -1);

                        if (flatBall != null) {
                            // Update trajectory history
                            posHistory.addLast(flatBall);
                            if (posHistory.size() > TRAJ_HISTORY) posHistory.removeFirst();

                            // Draw predicted trajectory arc on raw frame
                            drawTrajectory(frame);

                            boolean onTable =
                                flatBall.x >= 0 && flatBall.x <= FLAT_W &&
                                flatBall.y >= 0 && flatBall.y <= FLAT_H;

                            if (onTable) {
                                boolean onP1Side = flatBall.x < FLAT_W / 2.0;
                                String currentSide = onP1Side ? "P1" : "P2";

                                // ── Net crossing detection ────────────────
                                if (ballSeenOnce && onP1Side != wasOnP1Side) {
                                    stats.netCrossing();
                                    System.out.println("Net crossing! Rally count: " + stats.currentRally);
                                }
                                wasOnP1Side  = onP1Side;
                                ballSeenOnce = true;
                                lastSide     = currentSide;

                                // ── Bounce detection ──────────────────────
                                if (prevFlatY >= 0) {
                                    double dy = flatBall.y - prevFlatY;
                                    // Sign change: was going down, now going up = bounce
                                    if (prevDY > 1.5 && dy < -1.5) {
                                        if (bounceSide.equals(currentSide)) {
                                            bounceCount++;
                                        } else {
                                            bounceSide  = currentSide;
                                            bounceCount = 1;
                                        }
                                        System.out.printf("Bounce #%d on %s side%n",
                                            bounceCount, currentSide);

                                        // Two bounces on same side = point to opponent
                                        if (bounceCount >= BOUNCE_THRESHOLD &&
                                            now - lastPointTime > 1500) {
                                            String scorer = currentSide.equals("P1") ? "P2" : "P1";
                                            awardPoint(scorer);
                                            lastPointTime = now; pointMessageTime = now;
                                            resetBallTracking();
                                        }
                                    }
                                    prevDY = dy;
                                }
                                prevFlatY = flatBall.y;

                            } else {
                                // Ball left table boundary → opponent scores
                                if (!lastSide.isEmpty() && now - lastPointTime > 1500) {
                                    String scorer = lastSide.equals("P1") ? "P2" : "P1";
                                    awardPoint(scorer);
                                    lastPointTime = now; pointMessageTime = now;
                                    resetBallTracking();
                                }
                            }
                        }

                    } else {
                        // Ball not detected
                        ballMissingFrames++;

                        // Show predicted position as ghost circle
                        if (predictedPos != null) {
                            Point rawPredicted = toRawSpace(predictedPos);
                            if (rawPredicted != null) {
                                Imgproc.circle(frame, rawPredicted, (int) smoothedRadius,
                                    new Scalar(0, 200, 200), 1);  // cyan dashed ghost
                                drawText(frame, "?",
                                    (int) rawPredicted.x - 5,
                                    (int) rawPredicted.y + 5,
                                    0.5, new Scalar(0, 200, 200));
                            }
                        }

                        if (ballMissingFrames >= MISSING_THRESHOLD &&
                            !lastSide.isEmpty() && now - lastPointTime > 1500) {
                            String scorer = lastSide.equals("P1") ? "P2" : "P1";
                            awardPoint(scorer);
                            lastPointTime = now; pointMessageTime = now;
                            resetBallTracking();
                        }
                    }

                    // Win condition
                    if (scoreP1 >= 11 || scoreP2 >= 11) {
                        state = GameState.GAME_OVER;
                    }
                }

                // ── Scoreboard bar ─────────────────────────────────────────
                Imgproc.rectangle(frame, new Point(0, 0),
                    new Point(frame.cols(), 55), new Scalar(20, 20, 20), -1);

                String p1text = player1Name + ": " + scoreP1;
                String p2text = player2Name + ": " + scoreP2;

                drawText(frame, p1text, 10, 40, 1.0, new Scalar(100, 200, 255));
                drawText(frame, "VS", frame.cols() / 2 - 18, 38,
                    0.8, new Scalar(255, 255, 255));

                int[]  p2baseline = new int[1];
                Size   p2size     = Imgproc.getTextSize(p2text,
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, 2, p2baseline);
                drawText(frame, p2text,
                    (int)(frame.cols() - p2size.width - 10), 40,
                    1.0, new Scalar(100, 255, 100));

                // Rally counter
                drawText(frame,
                    "Rally: " + stats.currentRally,
                    frame.cols() / 2 - 40, 55,
                    0.5, new Scalar(200, 200, 200));

                // Flash point message
                long now2 = System.currentTimeMillis();
                if (!pointMessage.isEmpty() && now2 - pointMessageTime < 2000) {
                    drawText(frame, pointMessage,
                        frame.cols() / 2 - 162, frame.rows() / 2 + 2,
                        1.1, new Scalar(0, 0, 0));
                    drawText(frame, pointMessage,
                        frame.cols() / 2 - 160, frame.rows() / 2,
                        1.1, new Scalar(0, 255, 255));
                }

                // ── Game Over overlay + Stats ──────────────────────────────
                if (state == GameState.GAME_OVER) {
                    String winner = (scoreP1 > scoreP2 ? player1Name : player2Name) + " WINS! 🏆";

                    // Dark banner
                    Imgproc.rectangle(frame,
                        new Point(0, frame.rows() / 2 - 10),
                        new Point(frame.cols(), frame.rows()),
                        new Scalar(10, 10, 10), -1);

                    drawText(frame, winner,
                        frame.cols() / 2 - 210, frame.rows() / 2 + 35,
                        1.2, new Scalar(0, 215, 255));

                    // Stats lines
                    int sy = frame.rows() / 2 + 70;
                    drawText(frame,
                        "Total points: " + stats.totalPointsPlayed,
                        30, sy, 0.55, new Scalar(220, 220, 220));
                    drawText(frame,
                        "Longest rally: " + stats.longestRally + " crossings",
                        30, sy + 24, 0.55, new Scalar(220, 220, 220));
                    drawText(frame,
                        player1Name + " best streak: " + stats.p1MaxConsecutive,
                        30, sy + 48, 0.55, new Scalar(100, 200, 255));
                    drawText(frame,
                        player2Name + " best streak: " + stats.p2MaxConsecutive,
                        30, sy + 72, 0.55, new Scalar(100, 255, 100));
                    drawText(frame,
                        String.format("Avg rally length: %.1f crossings", stats.avgRallyLength()),
                        30, sy + 96, 0.55, new Scalar(200, 200, 100));
                    drawText(frame, "Press R to play again",
                        frame.cols() / 2 - 130, frame.rows() - 12,
                        0.65, new Scalar(200, 200, 200));
                }
            }

            // ── Render ────────────────────────────────────────────────────
            final ImageIcon icon = new ImageIcon(matToBufferedImage(frame));
            SwingUtilities.invokeLater(() -> {
                imageLabel.setIcon(icon);
                window.repaint();
            });

            Thread.sleep(30);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Homography ───────────────────────────────────────────────────────

    /**
     * Build the perspective transform matrix from the 4 clicked corners
     * to a canonical flat rectangle (FLAT_W × FLAT_H).
     * Corner order expected: TL, TR, BR, BL.
     */
    static void buildHomography() {
        // Source: the 4 clicked points in camera space
        MatOfPoint2f src = new MatOfPoint2f(
            tableCorners.get(0),   // TL
            tableCorners.get(1),   // TR
            tableCorners.get(2),   // BR
            tableCorners.get(3)    // BL
        );
        // Destination: flat rectangle
        MatOfPoint2f dst = new MatOfPoint2f(
            new Point(0,       0),
            new Point(FLAT_W,  0),
            new Point(FLAT_W,  FLAT_H),
            new Point(0,       FLAT_H)
        );
        homographyMatrix  = Imgproc.getPerspectiveTransform(src, dst);
        homographyInverse = Imgproc.getPerspectiveTransform(dst, src);
        System.out.println("Homography matrix built.");
    }

    /** Transform a point from camera space → flat table space. */
    static Point toFlatSpace(Point p) {
        if (homographyMatrix == null) return p;
        MatOfPoint2f src = new MatOfPoint2f(p);
        MatOfPoint2f dst = new MatOfPoint2f();
        Core.perspectiveTransform(src, dst, homographyMatrix);
        return dst.toArray()[0];
    }

    /** Transform a point from flat table space → camera space. */
    static Point toRawSpace(Point p) {
        if (homographyInverse == null) return p;
        MatOfPoint2f src = new MatOfPoint2f(p);
        MatOfPoint2f dst = new MatOfPoint2f();
        Core.perspectiveTransform(src, dst, homographyInverse);
        return dst.toArray()[0];
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Trajectory Prediction ────────────────────────────────────────────

    /**
     * Predict the next ball position by fitting a linear velocity to the
     * last few positions in flat space.
     * Returns null if there are fewer than 2 history points.
     */
    static Point predictNextPosition() {
        if (posHistory.size() < 2) return null;
        // Use the last two points for velocity
        Point[] arr = posHistory.toArray(new Point[0]);
        Point  last = arr[arr.length - 1];
        Point  prev = arr[arr.length - 2];
        double dx   = last.x - prev.x;
        double dy   = last.y - prev.y;
        return new Point(last.x + dx, last.y + dy);
    }

    /**
     * Draw the predicted trajectory as dots on the raw frame, converting
     * each predicted flat-space point back to camera space.
     */
    static void drawTrajectory(Mat frame) {
        if (posHistory.size() < 2) return;
        Point[] arr = posHistory.toArray(new Point[0]);
        Point  last = arr[arr.length - 1];
        Point  prev = arr[arr.length - 2];
        double dx   = last.x - prev.x;
        double dy   = last.y - prev.y;

        // Draw 4 future predicted positions
        for (int i = 1; i <= 4; i++) {
            Point flatPred = new Point(last.x + dx * i, last.y + dy * i);
            Point rawPred  = toRawSpace(flatPred);
            if (rawPred == null) continue;
            int alpha = 200 - i * 40;  // fade out
            Imgproc.circle(frame, rawPred, 3,
                new Scalar(0, alpha, alpha), -1);
        }

        // Draw history trail
        for (int i = 1; i < arr.length; i++) {
            Point rawA = toRawSpace(arr[i - 1]);
            Point rawB = toRawSpace(arr[i]);
            if (rawA == null || rawB == null) continue;
            Imgproc.line(frame, rawA, rawB,
                new Scalar(0, 180, 0), 1);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Scoring helpers ──────────────────────────────────────────────────

    static void awardPoint(String scorer) {
        if (scorer.equals("P1")) {
            scoreP1++;
            pointMessage = "Point → " + player1Name + "!";
        } else {
            scoreP2++;
            pointMessage = "Point → " + player2Name + "!";
        }
        stats.recordPoint(scorer);
        System.out.printf("Point! %s=%d  %s=%d%n",
            player1Name, scoreP1, player2Name, scoreP2);
    }

    static void resetBallTracking() {
        lastSide = "";
        smoothedCenter  = null;
        ballMissingFrames = 0;
        posHistory.clear();
        predictedPos = null;
        prevFlatY    = -1;
        prevDY       = 0;
        bounceSide   = "";
        bounceCount  = 0;
        ballSeenOnce = false;
    }

    static void resetGame() {
        scoreP1 = 0; scoreP2 = 0;
        pointMessage = "";
        stats.totalPointsPlayed  = 0;
        stats.longestRally       = 0;
        stats.currentRally       = 0;
        stats.p1MaxConsecutive   = 0;
        stats.p2MaxConsecutive   = 0;
        stats.p1Consecutive      = 0;
        stats.p2Consecutive      = 0;
        stats.totalRallyFrames   = 0;
        stats.rallyCount         = 0;
        resetBallTracking();
        state = GameState.PLAYING;
        System.out.println("🔄 Game restarted!");
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Ball detection ───────────────────────────────────────────────────

    static BallResult detectBall(Mat frame) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

        Mat mask = new Mat();
        Core.inRange(hsv, ballLower, ballUpper, mask);

        Mat kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.erode (mask, mask, kernel);
        Imgproc.dilate(mask, mask, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double  bestScore  = 0;
        Point   bestCenter = null;
        double  bestRadius = 0;
        float[] r = new float[1];
        Point   c = new Point();

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < 50 || area > 8000) continue;

            double perim = Imgproc.arcLength(
                new MatOfPoint2f(contour.toArray()), true);
            if (perim == 0) continue;

            double circ = (4 * Math.PI * area) / (perim * perim);
            if (circ < 0.55 || circ <= bestScore) continue;

            Imgproc.minEnclosingCircle(new MatOfPoint2f(contour.toArray()), c, r);

            if (registeredMinRadius > 0) {
                if (r[0] < registeredMinRadius || r[0] > registeredMaxRadius) continue;
            }

            bestScore  = circ;
            bestCenter = new Point(c.x, c.y);
            bestRadius = r[0];
        }

        return (bestCenter != null) ? new BallResult(bestCenter, bestRadius) : null;
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Utility ──────────────────────────────────────────────────────────

    static void drawText(Mat frame, String text, int x, int y,
                         double scale, Scalar color) {
        Imgproc.putText(frame, text, new Point(x, y),
            Imgproc.FONT_HERSHEY_SIMPLEX, scale, color, 2);
    }

    static Point midpoint(Point a, Point b) {
        return new Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
    }

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
