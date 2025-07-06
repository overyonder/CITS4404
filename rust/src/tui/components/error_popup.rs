//! The error popup component.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::Text,
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

/// Draws a popup box with an error message.
pub fn draw_error_popup(f: &mut Frame, error_message: &str, area: Rect) {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Min(10), // Minimum height for the popup
            Constraint::Percentage(33),
        ])
        .split(area);

    let popup_area = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(50), // Width of the popup
            Constraint::Percentage(25),
        ])
        .split(popup_layout[1])[1];

    let block = Block::default()
        .title("Error")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Red));

    let text = Text::from(format!(
        "{}\n\nPress any key to dismiss.",
        error_message
    ));
    let paragraph = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: true })
        .alignment(Alignment::Center);

    f.render_widget(Clear, popup_area); // This clears the area behind the popup
    f.render_widget(paragraph, popup_area);
}

