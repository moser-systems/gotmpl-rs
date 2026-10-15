//! Go-style context for execution errors.
//!
//! Go's `text/template` reports execution errors as
//! `template: <parse name>:<line>:<col>: executing "<name>" at <<node>>: <message>`,
//! where `<col>` is the 0-based byte offset within the line and `<node>` is the
//! failing parse node printed back as template source (`len 3`, `.A.B`,
//! `{{template "x"}}`). This module renders nodes the way Go's `Node.String`
//! does and computes the location from the template's source text.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::parse::{CommandNode, Expr, Number, PipeNode, Pos, TemplateNode};

/// Go's `%q` of a string.
fn go_quote(s: &str) -> String {
    let mut out = String::new();
    crate::go::go_quote_into(&mut out, s);
    out
}

/// Where a parsed template body came from: the name it was parsed under
/// (Go's `ParseName`) and the full source text it is a part of.
#[derive(Debug, Clone)]
pub(crate) struct TreeSource {
    pub(crate) parse_name: Option<String>,
    pub(crate) text: Arc<str>,
}

impl TreeSource {
    /// `<parse name>:<line>:<col>` for a byte offset, as Go's `ErrorContext`.
    pub(crate) fn location(&self, pos: Pos) -> String {
        let offset = pos.offset.min(self.text.len());
        let before = self.text.get(..offset).unwrap_or("");
        let col = before.rfind('\n').map_or(offset, |nl| offset - nl - 1);
        let line = 1 + before.matches('\n').count();
        format!("{}:{line}:{col}", self.parse_name.as_deref().unwrap_or(""))
    }

    /// The source text of the literal token starting at `offset`
    /// (Go keeps literals' original spelling, e.g. `1.0` or a raw string).
    fn literal_at(&self, offset: usize) -> Option<&str> {
        let rest = self.text.get(offset..)?;
        let bytes = rest.as_bytes();
        let end = match bytes.first()? {
            b'"' | b'\'' => {
                let quote = bytes[0];
                let mut i = 1;
                while i < bytes.len() && bytes[i] != quote {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
                (i + 1).min(bytes.len())
            }
            b'`' => rest[1..].find('`').map_or(bytes.len(), |i| i + 2),
            _ => {
                let mut i = 0;
                while i < bytes.len() {
                    let c = bytes[i];
                    let signed_exponent = (c == b'+' || c == b'-')
                        && (i == 0 || matches!(bytes[i - 1], b'e' | b'E' | b'p' | b'P'));
                    if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || signed_exponent {
                        i += 1;
                    } else {
                        break;
                    }
                }
                i
            }
        };
        rest.get(..end).filter(|s| !s.is_empty())
    }
}

/// Renders parse nodes as Go's `Node.String` does.
pub(crate) struct Renderer<'a> {
    pub(crate) source: Option<&'a TreeSource>,
}

impl Renderer<'_> {
    pub(crate) fn expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Dot(_) => ".".into(),
            Expr::Field(_, fields) => fields.iter().map(|f| format!(".{f}")).collect(),
            Expr::Variable(_, name, fields) => {
                let mut s = name.to_string();
                for f in fields {
                    s.push('.');
                    s.push_str(f);
                }
                s
            }
            Expr::Identifier(_, name) => name.to_string(),
            Expr::String(pos, value) => self.literal(*pos).unwrap_or_else(|| go_quote(value)),
            Expr::Number(pos, number) => self.literal(*pos).unwrap_or_else(|| match number {
                Number::Int(i) => i.to_string(),
                Number::Uint(u) => u.to_string(),
                Number::Float(f) => crate::value::Value::Float(*f).to_string(),
            }),
            Expr::Bool(_, b) => b.to_string(),
            Expr::Nil(_) => "nil".into(),
            Expr::Pipe(_, pipe) => format!("({})", self.pipe(pipe)),
            Expr::Chain(_, inner, fields) => {
                let mut s = self.expr(inner);
                for f in fields {
                    s.push('.');
                    s.push_str(f);
                }
                s
            }
        }
    }

    pub(crate) fn command(&self, cmd: &CommandNode) -> String {
        cmd.args
            .iter()
            .map(|a| self.expr(a))
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub(crate) fn pipe(&self, pipe: &PipeNode) -> String {
        let mut s = String::new();
        if !pipe.decl.is_empty() {
            s.push_str(
                &pipe
                    .decl
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            s.push_str(if pipe.is_assign { " = " } else { " := " });
        }
        s.push_str(
            &pipe
                .commands
                .iter()
                .map(|c| self.command(c))
                .collect::<Vec<_>>()
                .join(" | "),
        );
        s
    }

    pub(crate) fn template(&self, call: &TemplateNode) -> String {
        match &call.pipe {
            Some(pipe) => format!(
                "{{{{template {} {}}}}}",
                go_quote(&call.name),
                self.pipe(pipe)
            ),
            None => format!("{{{{template {}}}}}", go_quote(&call.name)),
        }
    }

    fn literal(&self, pos: Pos) -> Option<String> {
        self.source
            .and_then(|s| s.literal_at(pos.offset))
            .map(String::from)
    }
}

/// The position Go reports for an expression node. Go merges a field chain
/// into one node positioned at its *second* segment (`.A.B` reports `.B`),
/// and a variable with fields at its first field.
pub(crate) fn go_pos(expr: &Expr) -> Pos {
    match expr {
        Expr::Field(pos, fields) if fields.len() > 1 => {
            let first = fields.first().map_or(0, |f| f.len());
            Pos::new(pos.offset + 1 + first, pos.line)
        }
        Expr::Variable(pos, name, fields) if !fields.is_empty() => {
            Pos::new(pos.offset + name.len(), pos.line)
        }
        other => other.pos(),
    }
}

/// The position Go reports for a `{{template}}` call: its name token.
pub(crate) fn template_name_pos(call: &TemplateNode, source: Option<&TreeSource>) -> Pos {
    let Some(text) = source.map(|s| &s.text) else {
        return call.pos;
    };
    let rest = text.get(call.pos.offset..).unwrap_or("");
    let name_at = rest.find(['"', '`']).unwrap_or(0);
    Pos::new(call.pos.offset + name_at, call.pos.line)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(text: &str) -> TreeSource {
        TreeSource {
            parse_name: Some("t".into()),
            text: Arc::from(text),
        }
    }

    #[test]
    fn location_uses_zero_based_byte_columns() {
        let s = source("ab{{x}}\n  {{y}}");
        assert_eq!(s.location(Pos::new(2, 1)), "t:1:2");
        assert_eq!(s.location(Pos::new(12, 2)), "t:2:4");
        assert_eq!(s.location(Pos::new(999, 9)), "t:2:7");
        let unnamed = TreeSource {
            parse_name: None,
            text: Arc::from("{{x}}"),
        };
        assert_eq!(unnamed.location(Pos::new(2, 1)), ":1:2");
    }

    #[test]
    fn literals_keep_their_source_spelling() {
        let s = source(r#"{{eq 1.0 -2e-3 "a\"b" `raw` 'c' 0x1F}}"#);
        let lits: Vec<&str> = [5, 9, 15, 22, 28, 32]
            .iter()
            .filter_map(|&o| s.literal_at(o))
            .collect();
        assert_eq!(lits, ["1.0", "-2e-3", r#""a\"b""#, "`raw`", "'c'", "0x1F"]);
        assert_eq!(s.literal_at(999), None);
    }

    #[test]
    fn go_positions_for_chains() {
        let field = Expr::Field(Pos::new(2, 1), alloc::vec!["A".into(), "B".into()]);
        assert_eq!(go_pos(&field).offset, 4);
        let var = Expr::Variable(Pos::new(2, 1), "$x".into(), alloc::vec!["A".into()]);
        assert_eq!(go_pos(&var).offset, 4);
        let dot = Expr::Dot(Pos::new(7, 1));
        assert_eq!(go_pos(&dot).offset, 7);
    }
}
