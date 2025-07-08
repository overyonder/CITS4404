//! A reusable popup component for displaying messages.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame,
};

/// Draws a popup message box in the center of the screen.
///
/// # Arguments
/// * `f` - The `Frame` to draw on.
/// * `message` - The text to display in the popup.
/// * `title` - The title of the popup box.
/// * `color` - The color of the popup's border and title.
/// * `area` - The `Rect` to draw within.
pub fn draw_message_popup(
    f: &mut Frame,
    message: &str,
    title: &str,
    color: Color,
    area: Rect,
) {
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(color));

    let text = Paragraph::new(message)
        .block(block)
        .alignment(Alignment::Center);

    // Center the popup in the middle of the screen
    let popup_area = centered_rect(60, 25, area);
    f.render_widget(Clear, popup_area); // Clear the area behind the popup
    f.render_widget(text, popup_area);
}

/// Helper to create a centered rectangle for popups.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

