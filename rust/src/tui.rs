use crate::{
    constants::{LENGTH, MAX_POSITION, PADDLE_HEIGHT, PADDLE_WIDTH, WIDTH},
    game::GameState,
};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use std::io::{self, stdout};
use std::time::Duration;

/// Sets up the terminal for TUI rendering.
fn setup_terminal() -> io::Result<Terminal<impl Backend>> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout()))
}

/// Restores the terminal to its original state.
fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}

/// Renders a full game in the terminal using the TUI.
/// This function will take over the terminal and run a game loop.
/// It can optionally be passed a `GameState` to render. If `None`, a default state is created.
pub fn render_game(game_state_option: Option<GameState>) -> io::Result<()> {
    // Use the provided game state or create a new one.
    let mut game_state = game_state_option.unwrap_or_else(GameState::new);
    let mut terminal = setup_terminal()?;

    // Main game loop.
    loop {
        terminal.draw(|f| ui(f, &game_state))?;

        // Handle input: exit on 'q'.
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Char('q') = key.code {
                    break;
                }
            }
        }

        // Update the game state.
        game_state.tick();
    }

    restore_terminal()
}

/// Defines the UI layout and widgets for rendering the game state.
fn ui(frame: &mut Frame, game: &GameState) {
    // Create a layout to split the screen into a score area and a game area.
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // 1-line-high area for scores
            Constraint::Min(0),    // The rest of the screen for the game
        ])
        .split(frame.size());

    let score_area = main_layout[0];
    let game_area = main_layout[1];

    // Render the score paragraph.
    // NOTE: This assumes `game.score` is `[u32; 2]`. Adjust if this is incorrect.
    let score_text = format!("Score: {} - {}", game.score[0], game.score[1]);
    let score_paragraph = Paragraph::new(score_text).alignment(Alignment::Center);
    frame.render_widget(score_paragraph, score_area);

    // Render the game area block.
    let game_block = Block::default().borders(Borders::ALL).title("Pong");
    let game_area_inner = game_block.inner(game_area); // Get the inner rect to draw in.
    frame.render_widget(game_block, game_area);

    // Draw Paddles
    for (i, paddle) in game.paddles.iter().enumerate() {
        const PADDLE_WIDTH_CELLS: u16 = 1;
        let paddle_height_cells =
            (PADDLE_HEIGHT as f64 / LENGTH as f64 * game_area_inner.height as f64).round() as u16;

        let x_pos = if i == 0 {
            game_area_inner.left()
        } else {
            game_area_inner.right() - PADDLE_WIDTH_CELLS
        };

        // Normalize position against the maximum possible position.
        let y_fraction = paddle.position as f64 / MAX_POSITION as f64;
        let paddle_top_y =
            (y_fraction * (game_area_inner.height - paddle_height_cells) as f64).round() as u16;
        let y_pos = game_area_inner.y + paddle_top_y;

        let paddle_area = Rect::new(x_pos, y_pos, PADDLE_WIDTH_CELLS, paddle_height_cells);

        // Create a vertical bar of characters for the paddle's body.
        let paddle_body = "█\n".repeat(paddle_height_cells as usize);
        let paddle_paragraph = Paragraph::new(paddle_body);

        frame.render_widget(paddle_paragraph, paddle_area);
    }

    // Draw Ball
    // Convert ball's game coordinates to terminal cell coordinates.
    let ball_x_fraction = game.ball.position[0] / WIDTH as f64;
    let ball_y_fraction = game.ball.position[1] / LENGTH as f64;

    let ball_x = game_area_inner.x + (ball_x_fraction * game_area_inner.width as f64) as u16;
    let ball_y = game_area_inner.y + (ball_y_fraction * game_area_inner.height as f64) as u16;

    // Create a 1x1 Rect for the ball.
    let ball_rect = Rect::new(ball_x, ball_y, 1, 1);

    // Render the ball character.
    frame.render_widget(Paragraph::new("●"), ball_rect);
}
